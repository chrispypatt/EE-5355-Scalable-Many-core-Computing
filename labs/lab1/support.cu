/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void initVector(unsigned int **vec_h, unsigned int size, unsigned int max)
{
    *vec_h = (unsigned int*)malloc(size*sizeof(unsigned int));

    if(*vec_h == NULL) {
        FATAL("Unable to allocate host");
    }

    for (unsigned int i=0; i < size; i++) {
        (*vec_h)[i] = (rand()%(max + 1));
    }

}

void verify(unsigned int* in, unsigned int* out, unsigned int num_in, unsigned int num_out) {

  // Initialize reference
  unsigned int* out_ref = (unsigned int*) malloc(num_out*sizeof(unsigned int));
  for(unsigned int outIdx = 0; outIdx < num_out; ++outIdx) {
      out_ref[outIdx] = 0;
  }

  // Compute reference out
  for(unsigned int inIdx = 0; inIdx < num_in; ++inIdx) {
      unsigned int intermediate = outInvariant(in[inIdx]);
      for(unsigned int outIdx = 0; outIdx < num_out; ++outIdx) {
          out_ref[outIdx] += outDependent(intermediate, inIdx, outIdx);
      }
  }

  // Compare to reference out
  for(unsigned int outIdx = 0; outIdx < num_out; ++outIdx) {
      if(out[outIdx] != out_ref[outIdx]) {
        printf("TEST FAILED at output index %u, reference = %u, computed = %u"
          "\n\n", outIdx, out_ref[outIdx], out[outIdx]);
        exit(0);
      }
  }
  printf("TEST PASSED\n\n");

  free(out_ref);

}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

__host__ __device__ unsigned int outInvariant(unsigned int inValue) {
    return inValue*inValue;
}

__host__ __device__ unsigned int outDependent(unsigned int value, unsigned int inIdx,
        unsigned int outIdx) {
    if(inIdx == outIdx) {
        return 2*value;
    } else if(inIdx > outIdx) {
        return value/(inIdx - outIdx);
    } else {
        return value/(outIdx - inIdx);
    }
}

// Allocate a device array of same size as data.
unsigned int* allocateDeviceArray(unsigned int* data, int num_elements){
	int size = num_elements * sizeof(unsigned int);
	unsigned int* d_data = data;
	cudaError_t cuda_ret = cudaMalloc((void**) &d_data, size);
	if(cuda_ret != cudaSuccess) {
		printf("Unable to allocate device memory");
		exit(0);
	}
	return d_data;
}

// Copy a host array to a device array.
void copyToDeviceArray(unsigned int* d_data, const unsigned int*  h_data, int num_elements)
{
    int size = num_elements * sizeof(unsigned int);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
}

// Copy a device array to a host array.
void copyFromDeviceArray(unsigned int* h_data, const unsigned int*  d_data, int num_elements)
{
    int size = num_elements * sizeof(unsigned int);
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

}

