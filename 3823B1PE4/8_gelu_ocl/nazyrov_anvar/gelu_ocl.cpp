#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <stdexcept>

namespace {

const char* kKernelSource = R"CLC(
__kernel void gelu(__global const float* in, __global float* out, const int n) {
    const int i = get_global_id(0);
    if (i < n) {
        const float x = in[i];
        const float z = 1.5957691216f * (x + 0.044715f * x * x * x);
        out[i] = x / (1.0f + native_exp(-z));
    }
}
)CLC";

struct OclState {
    bool ready = false;
    int platform_index = -1;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem d_in = nullptr;
    cl_mem d_out = nullptr;
    size_t capacity = 0;
};

OclState g;

void Check(cl_int err, const char* what) {
    if (err != CL_SUCCESS) throw std::runtime_error(what);
}

void Init(int platform) {
    cl_uint num_platforms = 0;
    Check(clGetPlatformIDs(0, nullptr, &num_platforms), "clGetPlatformIDs");
    std::vector<cl_platform_id> platforms(num_platforms);
    Check(clGetPlatformIDs(num_platforms, platforms.data(), nullptr), "clGetPlatformIDs");
    if (platform < 0 || (cl_uint)platform >= num_platforms) throw std::runtime_error("bad platform");

    cl_device_id device = nullptr;
    Check(clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "clGetDeviceIDs");

    cl_int err = CL_SUCCESS;
    g.context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    Check(err, "clCreateContext");
    g.queue = clCreateCommandQueue(g.context, device, 0, &err);
    Check(err, "clCreateCommandQueue");

    g.program = clCreateProgramWithSource(g.context, 1, &kKernelSource, nullptr, &err);
    Check(err, "clCreateProgramWithSource");
    Check(clBuildProgram(g.program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr), "clBuildProgram");
    g.kernel = clCreateKernel(g.program, "gelu", &err);
    Check(err, "clCreateKernel");

    g.platform_index = platform;
    g.capacity = 0;
    g.d_in = g.d_out = nullptr;
    g.ready = true;
}

void EnsureCapacity(size_t n) {
    if (n <= g.capacity) return;
    if (g.d_in) clReleaseMemObject(g.d_in);
    if (g.d_out) clReleaseMemObject(g.d_out);
    cl_int err = CL_SUCCESS;
    g.d_in = clCreateBuffer(g.context, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer in");
    g.d_out = clCreateBuffer(g.context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, &err);
    Check(err, "clCreateBuffer out");
    g.capacity = n;
}

} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t n = input.size();
    std::vector<float> output(n);
    if (n == 0) return output;

    // Инициализация OpenCL выполняется один раз
    if (!g.ready || g.platform_index != platform) Init(platform);
    EnsureCapacity(n);

    Check(clEnqueueWriteBuffer(g.queue, g.d_in, CL_FALSE, 0, n * sizeof(float), input.data(), 0, nullptr, nullptr),
          "write");

    const int n_int = (int)n;
    Check(clSetKernelArg(g.kernel, 0, sizeof(cl_mem), &g.d_in), "arg0");
    Check(clSetKernelArg(g.kernel, 1, sizeof(cl_mem), &g.d_out), "arg1");
    Check(clSetKernelArg(g.kernel, 2, sizeof(int), &n_int), "arg2");

    const size_t local = 256;
    const size_t global = ((n + local - 1) / local) * local;
    Check(clEnqueueNDRangeKernel(g.queue, g.kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr), "kernel");

    Check(clEnqueueReadBuffer(g.queue, g.d_out, CL_TRUE, 0, n * sizeof(float), output.data(), 0, nullptr, nullptr),
          "read");
    return output;
}
