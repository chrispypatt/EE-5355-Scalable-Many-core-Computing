/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define TILE_SIZE 30

__global__ void kernel(int *A0, int *Anext, int nx, int ny, int nz) {
	#define A0(i, j, k) A0[((k)*ny + (j))*nx + (i)]
	#define Anext(i, j, k) Anext[((k)*ny + (j))*nx + (i)]
	
	int tx = threadIdx.x, ty = threadIdx.y;
	int dx = blockDim.x, dy = blockDim.y;
	int i = tx + dx * blockIdx.x;  
	int j = ty + dy * blockIdx.y;  

	//Load this thread's top, bottom, and center z values
	int bottom = A0(i,j,0);
	int center = A0(i,j,1);
	int top = A0(i,j,2);

	//create shared memory tile
	__shared__ int ds_A[TILE_SIZE][TILE_SIZE];

	//loop through all z slices
	for (int k=1; k<nz-1; ++k){
		//load up current z-axis slice
		ds_A[ty][tx] = center; 
		__syncthreads(); //wait for all other threads to do their thing

		Anext(i,j,k) = bottom + top +
			((tx > 0)?    ds_A[ty][tx-1]: (i==0)?    0: A0(i-1,j,k)) +
			((tx < dx-1)? ds_A[ty][tx+1]: (i==nx-1)? 0: A0(i+1,j,k)) +
			((ty > 0)?    ds_A[ty-1][tx]: 				A0(i,j-1,k)) +
			((ty < dy-1)? ds_A[ty+1][tx]: (j==ny-1)? 0: A0(i,j+1,k)) -
			6 * center;

		// Anext(i,j,k) = ((tx > 0)? ds_A[ty][tx-1]: (i==0)? 0:A0(i-1,j,k));
		// Anext(i,j,k) = ((tx < dx-1)? ds_A[ty][tx+1]: (i==nx-1)? 0:A0(i+1,j,k));
		// Anext(i,j,k) = 10*ty+j;//((ty > 0)?  ds_A[ty-1][tx]: (j==0)?  0:A0(i,j-1,k));
		// Anext(i,j,k) = ((ty < dy-1)? ds_A[ty+1][tx]: (j==ny-1)? 0:A0(i,j+1,k));
		// Anext(i,j,k) = bottom;
		// Anext(i,j,k) = top;
		// Anext(i,j,k) = 6*center;

		//shift z-values
		bottom = center; center = top;
		__syncthreads();
		//load new top value
		top = A0(i,j,k+2);
	}
	#undef A0
	#undef Anext
}

void launchStencil(int* A0, int* Anext, int nx, int ny, int nz) {
	//set-up space as TILE_SIZE=30 so each block is 30x30 threads
	//we need nx/30 x ny/30 blocks in each grid
    dim3 dimGrid(ceil(nx/double(TILE_SIZE)),ceil(ny/double(TILE_SIZE)),1);
	dim3 dimBlock(TILE_SIZE,TILE_SIZE,1);
	
	kernel<<<dimGrid, dimBlock>>>(A0,Anext,nx,ny,nz);
}

