#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

#include <vector>
#include <stdexcept>
#include <string>
#include <map>
#include <memory>
#include <mutex>
#include <limits>

// Проверка ошибок OpenCL
#define CL_CHECK(call) do {                           \
    cl_int err = (call);                              \
    if (err != CL_SUCCESS)                            \
        throw std::runtime_error(                     \
            "OpenCL error: " + std::to_string(err));  \
} while (0)

// Исходный код ядра OpenCL
static const char* KERNEL_SOURCE = R"CLC(
__kernel void gelu_kernel(
    __global const float* input,
    __global float* output,
    ulong n
) {
    size_t i = get_global_id(0);

    if ((ulong)i < n) {
        float x = input[i];
        float x3 = x * x * x;

        float z = -1.5957691216f *
                  (x + 0.044715f * x3);

        output[i] = x / (1.0f + exp(z));
    }
}
)CLC";

// Кэш контекста, программы и видеопамяти
struct OpenCLWorkspace {
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    cl_mem d_input = nullptr;
    cl_mem d_output = nullptr;
    size_t capacity = 0;

    OpenCLWorkspace(int platform_index) {
        try {
            cl_int err;

            // Получаем список платформ
            cl_uint count = 0;
            CL_CHECK(clGetPlatformIDs(0, nullptr, &count));

            if (platform_index < 0 ||
                static_cast<cl_uint>(platform_index) >= count) {
                throw std::invalid_argument(
                    "Invalid OpenCL platform"
                );
            }

            std::vector<cl_platform_id> platforms(count);
            CL_CHECK(clGetPlatformIDs(
                count, platforms.data(), nullptr
            ));

            cl_platform_id selected = platforms[platform_index];

            // Выбираем первое GPU-устройство платформы
            cl_uint device_count = 0;
            CL_CHECK(clGetDeviceIDs(
                selected, CL_DEVICE_TYPE_GPU,
                0, nullptr, &device_count
            ));

            if (device_count == 0) {
                throw std::runtime_error("No GPU device");
            }

            std::vector<cl_device_id> devices(device_count);
            CL_CHECK(clGetDeviceIDs(
                selected, CL_DEVICE_TYPE_GPU,
                device_count, devices.data(), nullptr
            ));

            cl_device_id device = devices[0];

            // Создаём контекст
            context = clCreateContext(
                nullptr, 1, &device,
                nullptr, nullptr, &err
            );
            CL_CHECK(err);

            // Создаём очередь команд
            queue = clCreateCommandQueue(
                context, device, 0, &err
            );
            CL_CHECK(err);

            // Компилируем программу один раз
            program = clCreateProgramWithSource(
                context, 1, &KERNEL_SOURCE,
                nullptr, &err
            );
            CL_CHECK(err);

            err = clBuildProgram(
                program, 1, &device,
                nullptr, nullptr, nullptr
            );

            if (err != CL_SUCCESS) {
                size_t log_size = 0;
                clGetProgramBuildInfo(
                    program, device, CL_PROGRAM_BUILD_LOG,
                    0, nullptr, &log_size
                );

                std::string log(log_size, '\0');

                if (log_size > 0) {
                    clGetProgramBuildInfo(
                        program, device, CL_PROGRAM_BUILD_LOG,
                        log_size, &log[0], nullptr
                    );
                }

                throw std::runtime_error(
                    "OpenCL build failed: " + log
                );
            }

            kernel = clCreateKernel(
                program, "gelu_kernel", &err
            );
            CL_CHECK(err);
        } catch (...) {
            cleanup();
            throw;
        }
    }

    // Выделяем память только при необходимости
    void reserve(size_t bytes) {
        if (bytes <= capacity) return;

        cl_int err;

        cl_mem new_input = clCreateBuffer(
            context, CL_MEM_READ_ONLY,
            bytes, nullptr, &err
        );
        CL_CHECK(err);

        cl_mem new_output = clCreateBuffer(
            context, CL_MEM_WRITE_ONLY,
            bytes, nullptr, &err
        );

        if (err != CL_SUCCESS) {
            clReleaseMemObject(new_input);
            CL_CHECK(err);
        }

        if (d_input) clReleaseMemObject(d_input);
        if (d_output) clReleaseMemObject(d_output);

        d_input = new_input;
        d_output = new_output;
        capacity = bytes;
    }

    void cleanup() {
        if (d_input) clReleaseMemObject(d_input);
        if (d_output) clReleaseMemObject(d_output);
        if (kernel) clReleaseKernel(kernel);
        if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    ~OpenCLWorkspace() {
        cleanup();
    }
};

// Интерфейс из задания
std::vector<float> GeluOCL(
    const std::vector<float>& input,
    int platform
) {
    std::vector<float> result(input.size());

    if (input.empty()) return result;

    if (input.size() >
        std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::length_error("Input too large");
    }

    // Отдельный кэш для каждой платформы
    static std::mutex mutex;
    static std::map<int, std::unique_ptr<OpenCLWorkspace>> cache;

    std::lock_guard<std::mutex> lock(mutex);

    auto& ws_ptr = cache[platform];

    if (!ws_ptr) {
        ws_ptr = std::make_unique<OpenCLWorkspace>(platform);
    }

    OpenCLWorkspace& ws = *ws_ptr;

    size_t bytes = input.size() * sizeof(float);
    ws.reserve(bytes);

    // CPU -> GPU
    CL_CHECK(clEnqueueWriteBuffer(
        ws.queue, ws.d_input, CL_TRUE,
        0, bytes, input.data(),
        0, nullptr, nullptr
    ));

    cl_ulong n = static_cast<cl_ulong>(input.size());

    CL_CHECK(clSetKernelArg(
        ws.kernel, 0, sizeof(cl_mem), &ws.d_input
    ));
    CL_CHECK(clSetKernelArg(
        ws.kernel, 1, sizeof(cl_mem), &ws.d_output
    ));
    CL_CHECK(clSetKernelArg(
        ws.kernel, 2, sizeof(cl_ulong), &n
    ));

    // Запускаем ядро
    const size_t local_size = 256;
    size_t global_size =
        ((input.size() + local_size - 1) / local_size)
        * local_size;

    CL_CHECK(clEnqueueNDRangeKernel(
        ws.queue, ws.kernel, 1,
        nullptr, &global_size, &local_size,
        0, nullptr, nullptr
    ));

    // GPU -> CPU (ожидаем завершения вычислений)
    CL_CHECK(clEnqueueReadBuffer(
        ws.queue, ws.d_output, CL_TRUE,
        0, bytes, result.data(),
        0, nullptr, nullptr
    ));

    return result;
}
