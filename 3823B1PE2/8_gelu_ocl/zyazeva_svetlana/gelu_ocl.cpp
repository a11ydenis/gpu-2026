#define CL_TARGET_OPENCL_VERSION 120

#include "gelu_ocl.h"
#include <CL/cl.h>

#include <string>
#include <limits>
#include <stdexcept>

namespace {

const char kernel_source[] = R"CLC(

__kernel void gelu_vec4(
    __global const float* input,
    __global float* output,
    int n4
) {
    int i = get_global_id(0);

    if (i < n4) {
        float4 x = vload4(i, input);
        float4 x3 = x * x * x;

        float4 inner = 0.7978845608f * (x + 0.044715f * x3);
        float4 e = exp(-2.0f * inner);
        float4 y = x / (1.0f + e);

        vstore4(y, i, output);
    }
}

__kernel void gelu_tail(
    __global const float* input,
    __global float* output,
    int start,
    int n
) {
    int i = get_global_id(0) + start;

    if (i < n) {
        float x = input[i];

        float inner = 0.7978845608f * (
            x + 0.044715f * x * x * x
        );

        float e = exp(-2.0f * inner);

        output[i] = x / (1.0f + e);
    }
}

)CLC";


struct OCLContext {
    int platform_index = -1;

    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;

    cl_kernel vec_kernel = nullptr;
    cl_kernel tail_kernel = nullptr;

    cl_mem d_input = nullptr;
    cl_mem d_output = nullptr;

    size_t capacity = 0;

    void ReleaseBuffers() {
        if (d_input) {
            clReleaseMemObject(d_input);
            d_input = nullptr;
        }

        if (d_output) {
            clReleaseMemObject(d_output);
            d_output = nullptr;
        }

        capacity = 0;
    }

    void Release() {
        ReleaseBuffers();

        if (vec_kernel) {
            clReleaseKernel(vec_kernel);
            vec_kernel = nullptr;
        }

        if (tail_kernel) {
            clReleaseKernel(tail_kernel);
            tail_kernel = nullptr;
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

        platform_index = -1;
    }

    void Init(int platform) {
        if (platform_index == platform && context != nullptr) {
            return;
        }

        Release();

        cl_uint count = 0;
        clGetPlatformIDs(0, nullptr, &count);

        if (platform < 0 || platform >= static_cast<int>(count)) {
            throw std::runtime_error("Invalid OpenCL platform");
        }

        std::vector<cl_platform_id> platforms(count);
        clGetPlatformIDs(count, platforms.data(), nullptr);

        cl_device_id device = nullptr;

        clGetDeviceIDs(
            platforms[platform],
            CL_DEVICE_TYPE_GPU,
            1,
            &device,
            nullptr
        );

        cl_int error;

        cl_context_properties props[] = {
            CL_CONTEXT_PLATFORM,
            reinterpret_cast<cl_context_properties>(platforms[platform]),
            0
        };

        context = clCreateContext(
            props, 1, &device, nullptr, nullptr, &error
        );

        queue = clCreateCommandQueue(
            context, device, 0, &error
        );

        const char* source = kernel_source;

        program = clCreateProgramWithSource(
            context, 1, &source, nullptr, &error
        );

        error = clBuildProgram(
            program, 1, &device, nullptr, nullptr, nullptr
        );

        vec_kernel = clCreateKernel(
            program, "gelu_vec4", &error
        );

        tail_kernel = clCreateKernel(
            program, "gelu_tail", &error
        );

        platform_index = platform;
    }

    void Reserve(size_t n) {
        if (n <= capacity) {
            return;
        }

        ReleaseBuffers();

        size_t bytes = n * sizeof(float);
        cl_int error;

        d_input = clCreateBuffer(
            context, CL_MEM_READ_ONLY, bytes, nullptr, &error
        );

        d_output = clCreateBuffer(
            context, CL_MEM_WRITE_ONLY, bytes, nullptr, &error
        );

        capacity = n;
    }

    ~OCLContext() {
        Release();
    }
};


OCLContext& GetContext(int platform) {
    static OCLContext ctx;
    ctx.Init(platform);
    return ctx;
}

}


std::vector<float> GeluOCL(
    const std::vector<float>& input,
    int platform
) {
    size_t n = input.size();

    if (n == 0) {
        return {};
    }

    if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Input is too large");
    }

    OCLContext& ctx = GetContext(platform);
    ctx.Reserve(n);

    int count = static_cast<int>(n);
    int n4 = count / 4;

    clEnqueueWriteBuffer(
        ctx.queue,
        ctx.d_input,
        CL_FALSE,
        0,
        n * sizeof(float),
        input.data(),
        0,
        nullptr,
        nullptr
    );

    if (n4 > 0) {
        clSetKernelArg(
            ctx.vec_kernel, 0, sizeof(cl_mem), &ctx.d_input
        );

        clSetKernelArg(
            ctx.vec_kernel, 1, sizeof(cl_mem), &ctx.d_output
        );

        clSetKernelArg(
            ctx.vec_kernel, 2, sizeof(int), &n4
        );

        size_t local = 256;
        size_t global = ((n4 + local - 1) / local) * local;

        clEnqueueNDRangeKernel(
            ctx.queue,
            ctx.vec_kernel,
            1,
            nullptr,
            &global,
            &local,
            0,
            nullptr,
            nullptr
        );
    }

    int rem = count % 4;

    if (rem > 0) {
        int start = n4 * 4;

        clSetKernelArg(
            ctx.tail_kernel, 0, sizeof(cl_mem), &ctx.d_input
        );

        clSetKernelArg(
            ctx.tail_kernel, 1, sizeof(cl_mem), &ctx.d_output
        );

        clSetKernelArg(
            ctx.tail_kernel, 2, sizeof(int), &start
        );

        clSetKernelArg(
            ctx.tail_kernel, 3, sizeof(int), &count
        );

        size_t global = rem;

        clEnqueueNDRangeKernel(
            ctx.queue,
            ctx.tail_kernel,
            1,
            nullptr,
            &global,
            nullptr,
            0,
            nullptr,
            nullptr
        );
    }

    std::vector<float> output(n);

    clEnqueueReadBuffer(
        ctx.queue,
        ctx.d_output,
        CL_TRUE,
        0,
        n * sizeof(float),
        output.data(),
        0,
        nullptr,
        nullptr
    );

    return output;
}
