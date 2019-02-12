/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"

int main (int argc, char *argv[])
{

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    int *A0_h, *Anext_h;
    int *A0_d, *Anext_d;
    size_t nx, ny, nz, size;

    if (argc == 1) {
        nx = 512;
        ny = 512;
        nz = 64;
    } else if (argc == 4) {
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
        nz = atoi(argv[3]);
    } else {
        printf("\n    Invalid input parameters!"
      "\n    Usage: ./stencil                 # Cube size = default (512x512x64)"
      "\n    Usage: ./stencil <NX> <NY> <NZ>  # Cube size = NX x NY x NZ"
      "\n");
        exit(0);
    }

    size = nx*ny*nz;

    A0_h = (int*) malloc( sizeof(int)*size );
    for (unsigned int i=0; i < size; i++) { A0_h[i] = rand()%100; }

    Anext_h = (int*) malloc( sizeof(int)*size );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Cube size: %u x %u x %u\n", nx, ny, nz);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMalloc((void**) &A0_d, sizeof(int)*size);
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**) &Anext_d, sizeof(int)*size);
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(A0_d, A0_h, sizeof(int)*size, cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    launchStencil(A0_d, Anext_d, nx, ny, nz);

    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);
    
    cuda_ret = cudaMemcpy(Anext_h, Anext_d, sizeof(int)*size, cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory from device");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(A0_h, Anext_h, nx, ny, nz);


    // Free memory ------------------------------------------------------------

    free(A0_h);
    free(Anext_h);

    cudaFree(A0_d);
    cudaFree(Anext_d);

    return 0;

}

