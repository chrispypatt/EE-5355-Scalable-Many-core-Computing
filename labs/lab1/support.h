/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#ifndef __FILEH__
#define __FILEH__

#include <sys/time.h>

typedef struct {
    struct timeval startTime;
    struct timeval endTime;
} Timer;

#ifdef __cplusplus
extern "C" {
#endif
void initVector(unsigned int **vec_h, unsigned int size, unsigned int max);
void verify(unsigned int* in, unsigned int* out, unsigned int num_in, unsigned int num_out);
void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);
void copyToDeviceArray(unsigned int* d_data, const unsigned int*  h_data, int num_elements);
void copyFromDeviceArray(unsigned int* h_data, const unsigned int*  d_data, int num_elements);
unsigned int* allocateDeviceArray(unsigned int* data, int num_elements);

__host__ __device__ unsigned int outInvariant(unsigned int inValue);
__host__ __device__ unsigned int outDependent(unsigned int value, unsigned int inIdx,
        unsigned int outIdx);

#ifdef __cplusplus
}
#endif

#define FATAL(msg, ...) \
    do {\
        fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__);\
        exit(-1);\
    } while(0)

#if __BYTE_ORDER != __LITTLE_ENDIAN
# error "File I/O is not implemented for this system: wrong endianness."
#endif

#endif
