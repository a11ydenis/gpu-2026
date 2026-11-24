#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif

#include "gelu_ocl.h"
#include <CL/cl.h>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>

namespace {
constexpr std::size_t kChunk = std::size_t(1) << 22;
const char kSource[] = R"CLC(
__kernel void gelu(__global float* data, const ulong count) {
    const size_t i = get_global_id(0) * 4;
    if (i >= count) {
        return;
    }

    if (count - i >= 4) {
        const float4 x = vload4(0, data + i);
        const float4 t = x * (1.5957691216057308f + 0.0713548162726009f * x * x);

        vstore4(x / (1.0f + native_exp(-t)), 0, data + i);
    } else {
        for (size_t j = i; j < count; ++j) {
            const float x = data[j];
            const float t = x * (1.5957691216057308f + 0.0713548162726009f * x * x);

            data[j] = x / (1.0f + native_exp(-t));
        }
    }
}
)CLC";

struct Slot {
    cl_mem device = nullptr;
    cl_mem host = nullptr;
    float* mapped = nullptr;
    cl_event ready = nullptr;

    std::size_t offset = 0;
    std::size_t count = 0;

    void Collect(float* output) {
        if (ready == nullptr) {
            return;
        }
        clWaitForEvents(1, &ready);
        std::memcpy(
            output + offset,
            mapped,
            count * sizeof(float));
        clReleaseEvent(ready);
        ready = nullptr;
    }
    void Release(cl_command_queue queue) {
        if (ready != nullptr) {
            clWaitForEvents(1, &ready);
            clReleaseEvent(ready);
            ready = nullptr;
        }
        if (mapped != nullptr) {
            clEnqueueUnmapMemObject(queue, host, mapped, 0, nullptr, nullptr);
            clFinish(queue);
            mapped = nullptr;
        }
        if (host != nullptr) {
            clReleaseMemObject(host);
            host = nullptr;
        }
        if (device != nullptr) {
            clReleaseMemObject(device);
            device = nullptr;
        }
    }
};

struct State {
    int platform_index;
    std::size_t capacity = 0;
    cl_context context = nullptr;
    cl_command_queue compute = nullptr;
    cl_command_queue download = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;

    Slot slots[2];

    explicit State(int platform) : platform_index(platform) {
        cl_uint count = 0;
        clGetPlatformIDs(0, nullptr, &count);
        std::vector<cl_platform_id> platforms(count);
        clGetPlatformIDs(count, platforms.data(), nullptr);
        cl_device_id device = nullptr;
        clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        compute = clCreateCommandQueue(context, device, 0, nullptr);
        download = clCreateCommandQueue(context, device, 0, nullptr);
        const char* source = kSource;
        program = clCreateProgramWithSource(context, 1, &source, nullptr, nullptr);

        clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        kernel = clCreateKernel(program, "gelu", nullptr);
    }

    ~State() {
        clFinish(compute);
        clFinish(download);

        for (Slot& slot : slots) {
            slot.Release(download);
        }

        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(compute);
        clReleaseCommandQueue(download);
        clReleaseContext(context);
    }

    void Reserve(std::size_t count) {
        const std::size_t required = std::min(count, kChunk);
        if (required <= capacity) {
            return;
        }
        const std::size_t bytes = required * sizeof(float);
        for (Slot& slot : slots) {
            slot.Release(download);
            slot.device = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, nullptr, nullptr);

            slot.host = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bytes, nullptr, nullptr);
            slot.mapped = static_cast<float*>(clEnqueueMapBuffer(download, slot.host, CL_TRUE, CL_MAP_READ, 0, bytes, 0, nullptr, nullptr, nullptr));
        }
        capacity = required;
    }
};

State& GetState(int platform) {
    static thread_local std::unique_ptr<State> state;

    if (!state || state->platform_index != platform) {
        state.reset(new State(platform));
    }

    return *state;
}

} // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    if (input.empty()) {
        return {};
    }

    State& state = GetState(platform);
    state.Reserve(input.size());
    std::vector<float> output(input.size());

    std::size_t offset = 0;
    std::size_t chunk = 0;

    while (offset < input.size()) {
        Slot& slot = state.slots[chunk % 2];
        slot.Collect(output.data());
        const std::size_t count =std::min(state.capacity, input.size() - offset);
        const std::size_t bytes = count * sizeof(float);
        const std::size_t global = count / 4 + (count % 4 != 0);
        const cl_ulong kernel_count = static_cast<cl_ulong>(count);
        slot.offset = offset;
        slot.count = count;

        
        clEnqueueWriteBuffer(state.compute, slot.device, CL_FALSE, 0, bytes, input.data() + offset, 0, nullptr, nullptr);
        clSetKernelArg(state.kernel, 0, sizeof(cl_mem), &slot.device);
        clSetKernelArg(state.kernel, 1, sizeof(kernel_count), &kernel_count);
        cl_event computed = nullptr;
        clEnqueueNDRangeKernel(state.compute, state.kernel,1, nullptr, &global, nullptr, 0, nullptr, &computed);
        clFlush(state.compute);

        clEnqueueReadBuffer(state.download, slot.device, CL_FALSE, 0, bytes, slot.mapped, 1, &computed, &slot.ready);
        clFlush(state.download);
        clReleaseEvent(computed);
        offset += count;
        ++chunk;
    }

    for (Slot& slot : state.slots) {
        slot.Collect(output.data());
    }

    return output;
}