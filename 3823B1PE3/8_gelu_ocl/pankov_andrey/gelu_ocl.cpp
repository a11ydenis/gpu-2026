#include "gelu_ocl.h"
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
void check(cl_int code, const char* operation) {
    if (code != CL_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(code));
}

template<class T, class F>
auto own(T object, F release, cl_int err, const char* operation) {
    check(err, operation);
    return std::unique_ptr<std::remove_pointer_t<T>, F>(object, release);
}

constexpr char source[] = R"CLC(
__kernel void gelu(__global float* values) {
    size_t i = get_global_id(0);
    float x = values[i];
    values[i] = 0.5f*x*(1.0f+tanh(0.7978845608f*x*(1.0f+0.044715f*x*x)));
}
)CLC";
}

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) return {};
    cl_uint count;
    check(clGetPlatformIDs(0, nullptr, &count), "count platforms");
    if (platform < 0 || static_cast<cl_uint>(platform) >= count)
        throw std::out_of_range("OpenCL platform");
    std::vector<cl_platform_id> platforms(count);
    check(clGetPlatformIDs(count, platforms.data(), nullptr), "list platforms");
    cl_device_id device;
    check(clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr), "select GPU");
    cl_int err;
    auto ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    auto context = own(ctx, clReleaseContext, err, "create context");
    auto q = clCreateCommandQueue(ctx, device, 0, &err);
    auto queue = own(q, clReleaseCommandQueue, err, "create queue");
    const char* code = source;
    auto prog = clCreateProgramWithSource(ctx, 1, &code, nullptr, &err);
    auto program = own(prog, clReleaseProgram, err, "create program");
    check(clBuildProgram(prog, 1, &device, nullptr, nullptr, nullptr), "build program");
    auto k = clCreateKernel(prog, "gelu", &err);
    auto kernel = own(k, clReleaseKernel, err, "create kernel");
    size_t size = input.size(), bytes = size*sizeof(float);
    auto mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bytes,
                              const_cast<float*>(input.data()), &err);
    auto buffer = own(mem, clReleaseMemObject, err, "allocate buffer");
    check(clSetKernelArg(k, 0, sizeof(mem), &mem), "set buffer");
    check(clEnqueueNDRangeKernel(q, k, 1, nullptr, &size, nullptr, 0, nullptr, nullptr), "enqueue GELU");
    std::vector<float> result(size);
    check(clEnqueueReadBuffer(q, mem, CL_TRUE, 0, bytes, result.data(), 0, nullptr, nullptr), "read result");
    return result;
}
