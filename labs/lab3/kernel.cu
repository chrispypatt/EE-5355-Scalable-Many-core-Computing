/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#define TILE_WIDTH_A 256
#define TILE_WIDTH_B 16
#define TILE_K (TILE_WIDTH_A/TILE_WIDTH_B)

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use register and shared memory tiling and thread coarsening
     *
     * NOTE: A and C are column major, B is row major
     *
     ********************************************************************/

    // Macros for accessing flattened matrices
    #define A(row,col) A[(row) + (col)*m]
    #define B(row,col) B[(row)*n + (col)]
    #define C(row,col) C[(row) + (col)*m]

    //tiling for B and output C
    __shared__ float shared_B[TILE_K][TILE_WIDTH_B];
    float C_RT[TILE_WIDTH_B];

    for(int i = 0; i<TILE_WIDTH_B;i++) C_RT[i] = 0.0;

    //Get block and thread idxs to load in tiles
    int by = blockIdx.y, bx = blockIdx.x, tx = threadIdx.x, ty = threadIdx.y;

    int b_col = tx + bx * TILE_WIDTH_B;
    int a_row = tx + ty * TILE_WIDTH_B + by * TILE_WIDTH_A;

    int p_col_offset = bx * TILE_WIDTH_B;

    for (int i = 0; i < ceil(double(k)/double(TILE_K)); i++){//loop through all k tiles
        //each thread load in an element of B Tile
        int b_row = i * TILE_K + ty;
        if (b_row < k && b_col < n){
            shared_B[ty][tx] = B(b_row,b_col);
        }else{
            shared_B[ty][tx] = 0;
        }
        __syncthreads();//wait for threads to load into shared mem
        for (int j = 0; j < TILE_K; j++){ 
            float a = 0;
            int a_col = i * TILE_K + j;
            if (a_col < k && a_row < m){
                a = A(a_row,a_col);
            }
            for (int l = 0; l < TILE_WIDTH_B; l++){//compute partial multiplication
                C_RT[l] += a*shared_B[j][l];
            }
        }
        __syncthreads();//wait for all threads to perform computations from b
    }
    for (int i = 0; i < TILE_WIDTH_B; i++) {
        if (a_row < m && i+p_col_offset < n){
            C(a_row,i+p_col_offset) = C_RT[i];
        }
    }
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'T') && (transb != 't')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------
    dim3 dimGrid(ceil(double(n)/double(TILE_WIDTH_B)),ceil(double(m)/double(TILE_K)),1);
    dim3 dimBlock(TILE_WIDTH_B,TILE_K,1);

    // Invoke CUDA kernel -----------------------------------------------------
    mysgemm<<<dimGrid, dimBlock>>>(m,n,k,A,B,C);
}


