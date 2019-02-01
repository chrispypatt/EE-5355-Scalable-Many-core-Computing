/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include "support.cu"

#define BLOCK_SIZE 1024

/******************************************************************************
 GPU kernels
*******************************************************************************/

__global__ void s2g_gpu_scatter_kernel(unsigned int* in, unsigned int* out,
    unsigned int num_in, unsigned int num_out) {

    int inIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (inIdx < num_in){//check that we are in array bounds
        unsigned int intermediate = outInvariant(in[inIdx]);
        for(unsigned int outIdx = 0; outIdx < num_out; ++outIdx) {
            atomicAdd(&out[outIdx], outDependent(intermediate, inIdx, outIdx));
        }
    }
}

__global__ void s2g_gpu_gather_kernel(unsigned int* in, unsigned int* out,
    unsigned int num_in, unsigned int num_out) {

    int outIdx = threadIdx.x + blockDim.x * blockIdx.x;    
    if (outIdx < num_out){//check that we are in array bounds
        for(unsigned int inIdx = 0; inIdx < num_in; ++inIdx) {
            unsigned int intermediate = outInvariant(in[inIdx]);
            out[outIdx] += outDependent(intermediate, inIdx, outIdx);
        }
    }
}

/******************************************************************************
 Scatter-to-gather functions
*******************************************************************************/
void s2g_cpu_scatter(unsigned int* in, unsigned int* out, unsigned int num_in,
    unsigned int num_out) {

    for(unsigned int inIdx = 0; inIdx < num_in; ++inIdx) {
        unsigned int intermediate = outInvariant(in[inIdx]);
        for(unsigned int outIdx = 0; outIdx < num_out; ++outIdx) {
            out[outIdx] += outDependent(intermediate, inIdx, outIdx);
        }
    }

}

void s2g_cpu_gather(unsigned int* in, unsigned int* out, unsigned int num_in,
    unsigned int num_out) {
    for(unsigned int outIdx = 0; outIdx < num_out; ++outIdx) {
        for(unsigned int inIdx = 0; inIdx < num_in; ++inIdx) {
            unsigned int intermediate = outInvariant(in[inIdx]);
            out[outIdx] += outDependent(intermediate, inIdx, outIdx);
        }
    }
}

void s2g_gpu_scatter(unsigned int* in, unsigned int* out, unsigned int num_in,
    unsigned int num_out) {

    //set-up space as BLOCK_SIZE=1024 threads per block 
    //and enough blocks to cover input space
    dim3 dimGrid(ceil(double(num_in)/double(BLOCK_SIZE)),1,1);
    dim3 dimBlock(BLOCK_SIZE,1,1);

    //run the scatter kernel
    s2g_gpu_scatter_kernel<<<dimGrid, dimBlock>>>(in,out,num_in,num_out);
}

void s2g_gpu_gather(unsigned int* in, unsigned int* out, unsigned int num_in,
    unsigned int num_out) {

    //set-up space as BLOCK_SIZE=1024 threads per block 
    //and enough blocks to cover output space
    dim3 dimGrid(ceil(double(num_out)/double(BLOCK_SIZE)),1,1);
    dim3 dimBlock(BLOCK_SIZE,1,1);

    //run the gather kernel
    s2g_gpu_gather_kernel<<<dimGrid, dimBlock>>>(in,out,num_in,num_out);
}


