#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

// OpenCL kernel for vector addition
const char *kernelSource = R"(
__kernel void vector_add(__global const float *A, __global const float *B, __global float *C, const int n) {
    int id = get_global_id(0);
    if (id < n) {
        C[id] = A[id] + B[id];
    }
}
)";

int main() {
    const int n = 1024;  // Number of elements in each vector
    std::vector<float> A(n, 1.0f), B(n, 2.0f), C(n, 0.0f);

    // Platform and device setup
    cl_platform_id platform;
    cl_device_id device;
    cl_int err = clGetPlatformIDs(1, &platform, nullptr);
    err |= clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    // Buffers
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), A.data(), &err);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), B.data(), &err);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, &err);

    // Build the program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    // Create the kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &n);

    // Execute the kernel
    size_t globalSize = n;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);

    // Read back the result
    clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, n * sizeof(float), C.data(), 0, nullptr, nullptr);

    // Check the result
    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (C[i] != A[i] + B[i]) {
            success = false;
            break;
        }
    }
    std::cout << "Result is " << (success ? "correct" : "incorrect") << std::endl;

    // Cleanup
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
