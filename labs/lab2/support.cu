/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include "support.h"

void verify(int *A0, int *Anext, unsigned int nx, unsigned int ny,
  unsigned int nz) {

  #define A0(i, j, k) A0[((k)*ny + (j))*nx + (i)]
  #define Anext(i, j, k) Anext[((k)*ny + (j))*nx + (i)]

  for(int k = 1; k < nz - 1; ++k) {
    for(int j = 1; j < ny - 1; ++j) {
      for(int i = 1; i < nx - 1; ++i) {
        int result = 
            A0(i    , j    , k + 1)
          + A0(i    , j    , k - 1)
          + A0(i    , j + 1, k    )
          + A0(i    , j - 1, k    )
          + A0(i + 1, j    , k    )
          + A0(i - 1, j    , k    )
          - 6*A0(i    , j    , k    );
        if (result != Anext(i, j, k) ) {
          printf("TEST FAILED at (%d, %d, %d): GPU = %d, CPU = %d\n\n", i, j, k, Anext(i, j, k), result);
          exit(0);
        }
      }
    }
  }
  printf("TEST PASSED\n\n");
  #undef A0
  #undef Anext

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

