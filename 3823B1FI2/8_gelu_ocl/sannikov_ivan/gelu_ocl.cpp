#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif

#include "gelu_ocl.h"
#include <CL/cl.h>

#include <cstddef>

namespace {

constexpr std::size_t kLocalSize = 256;

const char kSource[] = R"CLC(
__kernel void gelu_vec4(
    __global const float4* restrict input,
    __global float4* restrict output,
    const ulong count)
{
    const size_t i = get_global_id(0);

    if (i < count) {
        const float4 x = input[i];
        const float4 x2 = x * x;
        const float4 z =
            -1.595769122f * x * mad(0.044715f, x2, 1.0f);

        output[i] = x * native_recip(1.0f + native_exp(z));
    }
}

__kernel void gelu_scalar(
    __global const float* restrict input,
    __global float* restrict output,
    const ulong offset,
    const ulong count)
{
    const size_t i = offset + get_global_id(0);

    if (i < count) {
        const float x = input[i];
        const float x2 = x * x;
        const float z =
            -1.595769122f * x * mad(0.044715f, x2, 1.0f);

        output[i] = x * native_recip(1.0f + native_exp(z));
    }
}
)CLC";

struct State {
    int platform_index = -1;
    std::size_t capacity = 0;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel_vec4 = nullptr;
    cl_kernel kernel_scalar = nullptr;
    cl_mem input_buffer = nullptr;
    cl_mem output_buffer = nullptr;

    State() = default;
    State(const State&) = delete;
    State& operator=(const State&) = delete;

    ~State() {
        Release();
    }

    void Release() {
        if (input_buffer) {
            clReleaseMemObject(input_buffer);
            input_buffer = nullptr;
        }

        if (output_buffer) {
            clReleaseMemObject(output_buffer);
            output_buffer = nullptr;
        }

        if (kernel_vec4) {
            clReleaseKernel(kernel_vec4);
            kernel_vec4 = nullptr;
        }

        if (kernel_scalar) {
            clReleaseKernel(kernel_scalar);
            kernel_scalar = nullptr;
        }

        if (program) {
            clReleaseProgram(program);
            program = nullptr;
        }

        if (queue) {
            clReleaseCommandQueue(queue);
            queue = nullptr;
        }

        if (context) {
            clReleaseContext(context);
            context = nullptr;
        }

        capacity = 0;
        platform_index = -1;
    }

    void Init(int platform) {
        if (context && platform_index == platform) {
            return;
        }

        Release();
        cl_uint platform_count = 0;
        clGetPlatformIDs(0, nullptr, &platform_count);
        std::vector<cl_platform_id> platforms(platform_count);
        clGetPlatformIDs(platform_count, platforms.data(), nullptr);
        cl_device_id device = nullptr;
        clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        queue = clCreateCommandQueue(context, device, 0, nullptr);
        const char* source = kSource;
        program = clCreateProgramWithSource(context, 1, &source, nullptr, nullptr);
        clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math -cl-mad-enable", nullptr, nullptr);
        kernel_vec4 = clCreateKernel(program, "gelu_vec4", nullptr);
        kernel_scalar = clCreateKernel(program, "gelu_scalar", nullptr);
        platform_index = platform;
    }

    void Reserve(std::size_t count) {
        if (count <= capacity) {
            return;
        }

        if (input_buffer) {
            clReleaseMemObject(input_buffer);
        }

        if (output_buffer) {
            clReleaseMemObject(output_buffer);
        }

        const std::size_t bytes = count * sizeof(float);

        input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
        output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);
        capacity = count;
    }
};

State& GetState(int platform) {
    static thread_local State state;
    state.Init(platform);
    return state;
}

} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform)
{
    if (input.empty()) {
        return {};
    }

    State& state = GetState(platform);
    const std::size_t n = input.size();
    const std::size_t bytes = n * sizeof(float);
    state.Reserve(n);
    clEnqueueWriteBuffer(state.queue, state.input_buffer, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr);
    const cl_ulong vector_count = static_cast<cl_ulong>(n / 4);

    if (vector_count > 0) {
        const std::size_t global = ((n / 4 + kLocalSize - 1) / kLocalSize) * kLocalSize;
        clSetKernelArg(state.kernel_vec4, 0, sizeof(cl_mem), &state.input_buffer);
        clSetKernelArg(state.kernel_vec4, 1, sizeof(cl_mem), &state.output_buffer);
        clSetKernelArg(state.kernel_vec4, 2, sizeof(vector_count), &vector_count);

        clEnqueueNDRangeKernel(state.queue, state.kernel_vec4, 1, nullptr, &global, &kLocalSize, 0, nullptr, nullptr);
    }

    const std::size_t tail = n % 4;

    if (tail > 0) {
        const cl_ulong offset = static_cast<cl_ulong>(n - tail);
        const cl_ulong count = static_cast<cl_ulong>(n);
        clSetKernelArg(state.kernel_scalar, 0, sizeof(cl_mem), &state.input_buffer);
        clSetKernelArg(state.kernel_scalar, 1, sizeof(cl_mem), &state.output_buffer);
        clSetKernelArg(state.kernel_scalar, 2, sizeof(offset), &offset);
        clSetKernelArg(state.kernel_scalar, 3, sizeof(count), &count);
        clEnqueueNDRangeKernel(state.queue,state.kernel_scalar, 1, nullptr, &tail, nullptr, 0, nullptr, nullptr);
    }

    clFlush(state.queue);
    std::vector<float> output(n);
    clEnqueueReadBuffer(state.queue, state.output_buffer, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);


    
    return output;
}