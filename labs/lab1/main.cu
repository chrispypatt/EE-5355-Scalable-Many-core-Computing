/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#include "kernel.cu"

int main(int argc, char* argv[])
{
    Timer timer;

    // Initialize host variables ----------------------------------------------

    unsigned int *in_h;
    unsigned int *out_h;
    unsigned int *in_d;
    unsigned int *out_d;
    unsigned int num_in, num_out;
    cudaError_t cuda_ret;

    enum Mode {CPU_SCATTER = 1, CPU_GATHER, GPU_SCATTER, GPU_GATHER};
    Mode mode;

    if(argc == 2) {
        mode = (Mode) atoi(argv[1]);
        num_in = num_out = 4096;
    } else if(argc == 3) {
        mode = (Mode) atoi(argv[1]);
        num_in = num_out = atoi(argv[2]);
    } else if(argc == 4) {
        mode = (Mode) atoi(argv[1]);
        num_in = atoi(argv[2]);
        num_out = atoi(argv[3]);
    } else {
        printf("\n    Invalid input parameters."
        "\n"
        "\n    Usage: ./s2g <m>          # Mode: m, Input: 4,096, Output: 4,096"
        "\n           ./s2g <m> <M>      # Mode: m, Input:     M, Output:     M"
        "\n           ./s2g <m> <M> <N>  # Mode: m, Input:     M, Output:     N"
        "\n"
        "\n    Modes: 1 = CPU Scatter"
        "\n           2 = CPU Gather"
        "\n           3 = GPU Scatter"
        "\n           4 = GPU Gather"
        "\n\n");
        exit(0);
    }

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    initVector(&in_h, num_in, 100);
    out_h = (unsigned int*) malloc(num_out*sizeof(unsigned int));
    memset((void*) out_h, 0, num_out*sizeof(unsigned int));

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Input size = %u\n    Output size = %u\n", num_in, num_out);

    // Allocate device variables ----------------------------------------------

    if(mode == GPU_SCATTER || mode == GPU_GATHER) {

        printf("Allocating device variables..."); fflush(stdout);
        startTime(&timer);

        cuda_ret = cudaMalloc((void**)&in_d, num_in * sizeof(unsigned int));
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
        cuda_ret = cudaMalloc((void**)&out_d, num_out * sizeof(unsigned int));
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    }

    // Copy host variables to device ------------------------------------------

    if(mode == GPU_SCATTER || mode == GPU_GATHER) {

        printf("Copying data from host to device..."); fflush(stdout);
        startTime(&timer);

        cuda_ret = cudaMemcpy(in_d, in_h, num_in * sizeof(unsigned int),
            cudaMemcpyHostToDevice);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

        cuda_ret = cudaMemset(out_d, 0, num_out * sizeof(unsigned int));
        if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");

        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    }

    // Launch kernel ----------------------------------------------------------

    printf("Launching kernel ");

    if(mode == CPU_SCATTER) {
        printf("(CPU scatter version)...");fflush(stdout);
        startTime(&timer);
        s2g_cpu_scatter(in_h, out_h, num_in, num_out);
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    } else if(mode == CPU_GATHER) {
        printf("(CPU gather version)...");fflush(stdout);
        startTime(&timer);
        s2g_cpu_gather(in_h, out_h, num_in, num_out);
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    } else if(mode == GPU_SCATTER) {
        printf("(GPU scatter version)...");fflush(stdout);
        startTime(&timer);
        s2g_gpu_scatter(in_d, out_d, num_in, num_out);
        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    } else if(mode == GPU_GATHER) {
        printf("(GPU gather version)...");fflush(stdout);
        startTime(&timer);
        s2g_gpu_gather(in_d, out_d, num_in, num_out);
        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    } else {
        printf("Invalid mode!\n");
        exit(0);
    }


    // Copy device variables from host ----------------------------------------

    if(mode == GPU_SCATTER || mode == GPU_GATHER) {

        printf("Copying data from device to host..."); fflush(stdout);
        startTime(&timer);

        cuda_ret = cudaMemcpy(out_h, out_d, num_out * sizeof(unsigned int),
            cudaMemcpyDeviceToHost);
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

        cudaDeviceSynchronize();
        stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    }

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(in_h, out_h, num_in, num_out);

    // Free memory ------------------------------------------------------------

    free(in_h); free(out_h);
    if(mode == GPU_SCATTER || mode == GPU_GATHER) {
        cudaFree(in_d); cudaFree(out_d);
    }

    return 0;
}

