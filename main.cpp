#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define DATA_SIZE 1024

int main() {
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    // Get the default OpenCL platform
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to get platform IDs: %d\n", err);
        return 1;
    }

    // Get the default OpenCL device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to get device IDs: %d\n", err);
        return 1;
    }

    // Create a context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create context: %d\n", err);
        return 1;
    }

    // Create a command queue
    queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create command queue: %d\n", err);
        return 1;
    }

    // Create the kernel program from the source code
    const char* kernel_source =
            "__kernel void vector_add(__global float* a, __global float* b, __global float* c) {\n"
            "    int i = get_global_id(0);\n"
            "    c[i] = a[i] + b[i];\n"
            "}\n";

    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create program: %d\n", err);
        return 1;
    }

    // Build the program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to build program: %d\n", err);
        return 1;
    }

    // Create the kernel
    kernel = clCreateKernel(program, "vector_add", &err);
    if (err != CL_SUCCESS) {
        printf("Failed to create kernel: %d\n", err);
        return 1;
    }

    // Allocate memory on the device
    cl_mem a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * DATA_SIZE, NULL, &err);
    cl_mem b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * DATA_SIZE, NULL, &err);
    cl_mem c_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * DATA_SIZE, NULL, &err);

    // Set the kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buf);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buf);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buf);

    // Enqueue the kernel
    size_t global_size = DATA_SIZE;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to enqueue kernel: %d\n", err);
        return 1;
    }

    // Wait for the kernel to finish
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        printf("Failed to wait for kernel: %d\n", err);
        return 1;
    }

    // Read the results back from the device
    float* c = (float*)malloc(sizeof(float) * DATA_SIZE);
    err = clEnqueueReadBuffer(queue, c_buf, CL_TRUE, 0, sizeof(float) * DATA_SIZE, c, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to read buffer: %d\n", err);
        return 1;
    }

    // Print the first few results
    printf("Results:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", c[i]);
    }
    printf("\n");

    // Clean up
    free(c);
    clReleaseMemObject(a_buf);
    clReleaseMemObject(b_buf);
    clReleaseMemObject(c_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}