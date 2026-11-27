#include "gelu_ocl.h"
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
const char* source = R"(
__kernel void gelu(__global const float* x, __global float* y, ulong n) {
    ulong i = get_global_id(0);
    if (i < n) {
        float value = x[i];
        float z = 0.7978845608f * (value + 0.044715f * value * value * value);
        y[i] = 0.5f * value * (1.0f + tanh(z));
    }
})";
void Check(cl_int error) { if (error != CL_SUCCESS) throw std::runtime_error("OpenCL error " + std::to_string(error)); }
}
std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    std::vector<float> output(input.size());
    if (input.empty()) return output;
    cl_uint count = 0;
    Check(clGetPlatformIDs(0, nullptr, &count));
    if (platform < 0 || static_cast<cl_uint>(platform) >= count) throw std::invalid_argument("invalid platform");
    std::vector<cl_platform_id> platforms(count);
    Check(clGetPlatformIDs(count, platforms.data(), nullptr));
    cl_device_id device;
    Check(clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr));
    cl_int error;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    cl_mem in = nullptr, out = nullptr;
    try {
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error); Check(error);
        queue = clCreateCommandQueue(context, device, 0, &error); Check(error);
        program = clCreateProgramWithSource(context, 1, &source, nullptr, &error); Check(error);
        error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (error != CL_SUCCESS) {
            size_t length = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &length);
            std::string log(length, '\0');
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, length, log.data(), nullptr);
            throw std::runtime_error(log);
        }
        kernel = clCreateKernel(program, "gelu", &error); Check(error);
        size_t bytes = input.size() * sizeof(float);
        in = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, nullptr, &error); Check(error);
        out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, nullptr, &error); Check(error);
        Check(clEnqueueWriteBuffer(queue, in, CL_TRUE, 0, bytes, input.data(), 0, nullptr, nullptr));
        cl_ulong n = input.size();
        Check(clSetKernelArg(kernel, 0, sizeof(in), &in));
        Check(clSetKernelArg(kernel, 1, sizeof(out), &out));
        Check(clSetKernelArg(kernel, 2, sizeof(n), &n));
        size_t global = input.size();
        Check(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global, nullptr, 0, nullptr, nullptr));
        Check(clEnqueueReadBuffer(queue, out, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr));
    } catch (...) {
        if (out) clReleaseMemObject(out); if (in) clReleaseMemObject(in);
        if (kernel) clReleaseKernel(kernel); if (program) clReleaseProgram(program);
        if (queue) clReleaseCommandQueue(queue); if (context) clReleaseContext(context);
        throw;
    }
    clReleaseMemObject(out); clReleaseMemObject(in); clReleaseKernel(kernel);
    clReleaseProgram(program); clReleaseCommandQueue(queue); clReleaseContext(context);
    return output;
}
