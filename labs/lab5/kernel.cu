/******************************************************************************
 *cr
 *cr         (C) Copyright 2010-2013 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512
#define WARP_SIZE 32
#define NUM_WARPS (BLOCK_SIZE/WARP_SIZE)

// Maximum number of elements that can be inserted into a block queue
#define BQ_CAPACITY 2048

// Maximum number of elements that can be inserted into a warp queue
#define WQ_CAPACITY 128

/******************************************************************************
 GPU kernels
*******************************************************************************/

__global__ void gpu_global_queuing_kernel(unsigned int *nodePtrs,
  unsigned int *nodeNeighbors, unsigned int *nodeVisited,
  unsigned int *currLevelNodes, unsigned int *nextLevelNodes,
  unsigned int *numCurrLevelNodes, unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  //large number of nodes is too many for our threads so each thread may have to do more than one
  for (idx=idx; idx < *numCurrLevelNodes; idx += gridDim.x * blockDim.x){
    unsigned int node = currLevelNodes[idx];
    for(unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1]; ++nbrIdx) {
      //check if node was visited, if it wasn't, flag it as visited and update queue
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      unsigned int visited = atomicAdd(&(nodeVisited[neighbor]), 1); 
      if(!visited){
        //increment numNextLevelNodes and use old value as index for this node's place in the queue
        unsigned int gq_idx = atomicAdd(numNextLevelNodes,1);
        nextLevelNodes[gq_idx] = neighbor;
      }
    }
  }
}

__global__ void gpu_block_queuing_kernel(unsigned int *nodePtrs,
  unsigned int *nodeNeighbors, unsigned int *nodeVisited,
  unsigned int *currLevelNodes, unsigned int *nextLevelNodes,
  unsigned int *numCurrLevelNodes, unsigned int *numNextLevelNodes) {
  
  //setup block's shared queue
  __shared__ unsigned int s_nextLevelNodes[BQ_CAPACITY], s_numNextLevelNodes, s_start;

  if (threadIdx.x == 0) s_numNextLevelNodes = 0; //init block's numNExtLevelNodes
  __syncthreads();

  // INSERT KERNEL CODE HERE
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  //large number of nodes is too many for our threads so each thread may have to do more than one
  for (idx=idx; idx < *numCurrLevelNodes; idx += gridDim.x * blockDim.x){
    unsigned int node = currLevelNodes[idx];
    for(unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1]; ++nbrIdx) {
      //check if node was visited, if it wasn't, flag it as visited and update queue
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      unsigned int visited = atomicAdd(&(nodeVisited[neighbor]), 1); 
      if(!visited){
        //increment numNextLevelNodes and use old value as index for this node's place in the queue
        unsigned int bq_idx = atomicAdd(&s_numNextLevelNodes,1);
        if (bq_idx < BQ_CAPACITY){//make sure there is room in block queue
          s_nextLevelNodes[bq_idx] = neighbor;
        }else{//if not, put right into global queue
          s_numNextLevelNodes = BQ_CAPACITY;//s_numNextLevelNodes >= BQ_CAPACITY so reset to BQ_CAPACITY
          unsigned int gq_idx = atomicAdd(numNextLevelNodes,1);
          nextLevelNodes[gq_idx] = neighbor;
        }
      }
    }
  }
  __syncthreads();//wait for entire block to finish

  //update global numNextLevelNodes for other blocks to determine their start 
  if (threadIdx.x == 0){
    s_start = atomicAdd(numNextLevelNodes, s_numNextLevelNodes);
  }
  __syncthreads();

  for (unsigned int i = threadIdx.x; i < s_numNextLevelNodes; i += blockDim.x){
    nextLevelNodes[i+s_start] = s_nextLevelNodes[i];
  }
}



__global__ void gpu_warp_queuing_kernel(unsigned int *nodePtrs,
  unsigned int *nodeNeighbors, unsigned int *nodeVisited,
  unsigned int *currLevelNodes, unsigned int *nextLevelNodes,
  unsigned int *numCurrLevelNodes, unsigned int *numNextLevelNodes) {

  // INSERT KERNEL CODE HERE
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;

  //setup block's shared queue
  __shared__ unsigned int b_nextLevelNodes[BQ_CAPACITY], b_numNextLevelNodes, b_start;

  //setup warp queues

  unsigned int wqueue_idx = threadIdx.x % WARP_SIZE;
  __shared__ unsigned int w_nextLevelNodes[WQ_CAPACITY][WARP_SIZE];//allows for coalescing later
  __shared__ unsigned int w_numNextLevelNodes[WARP_SIZE], w_start[WARP_SIZE];

  //init block's numNextLevelNodes
  if (threadIdx.x == 0) b_numNextLevelNodes = 0; 
  //init each warp queue's numNextLevelNodes
  if (threadIdx.x < WARP_SIZE) {
    w_numNextLevelNodes[threadIdx.x] = 0; 
  } 
  __syncthreads();

  //large number of nodes is too many for our threads so each thread may have to do more than one
  for (idx=idx; idx < *numCurrLevelNodes; idx += gridDim.x * blockDim.x){
    unsigned int node = currLevelNodes[idx];
    for(unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1]; ++nbrIdx) {
      //check if node was visited, if it wasn't, flag it as visited and update queue
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      unsigned int visited = atomicAdd(&(nodeVisited[neighbor]), 1); 
      if(!visited){
        //increment warp level numNextLevelNodes and use old value as index for this node's place in the queue
        unsigned int queue_idx = atomicAdd(&(w_numNextLevelNodes[wqueue_idx]),1);
        if(queue_idx < WQ_CAPACITY){//make sure there is room in this thread's warp queue
          w_nextLevelNodes[queue_idx][wqueue_idx] = neighbor;
        }else{//if not, fall back to block and global queues
          //increment block level numNextLevelNodes and use old value as index for this node's place in the queue
          w_numNextLevelNodes[wqueue_idx] = BQ_CAPACITY;//w_numNextLevelNodes[wqueue_idx] >= WQ_CAPACITY so reset to WQ_CAPACITY
          unsigned int bq_idx = atomicAdd(&b_numNextLevelNodes,1);
          if (bq_idx < BQ_CAPACITY){//make sure there is room in block queue
            b_nextLevelNodes[bq_idx] = neighbor;
          }else{//if not, put right into global queue
            b_numNextLevelNodes = BQ_CAPACITY;//s_numNextLevelNodes >= BQ_CAPACITY so reset to BQ_CAPACITY
            unsigned int gq_idx = atomicAdd(numNextLevelNodes,1);
            nextLevelNodes[gq_idx] = neighbor;
          }
        }
      }
    }
  }
  __syncthreads();//wait for entire block to finish

  //update block's numNextLevelNodes so other warps can determine their start
  unsigned int offset = threadIdx.x/WARP_SIZE;
  if (offset == 0){//only first thread in a warp 
    w_start[wqueue_idx] = atomicAdd(&b_numNextLevelNodes, w_numNextLevelNodes[wqueue_idx]);
  }
  __syncthreads();
  //let each thread in the warp move elements from warp queue to block queue in coalesced fashion
  for (unsigned int i = offset; i < w_numNextLevelNodes[wqueue_idx]; i += NUM_WARPS){
    unsigned int bq_idx = w_start[wqueue_idx] + i;
    if (bq_idx < BQ_CAPACITY){//make sure there is room in block queue
      b_nextLevelNodes[bq_idx] = w_nextLevelNodes[i][wqueue_idx];
    }else{//if not, put right into global queue
      b_numNextLevelNodes = BQ_CAPACITY;//s_numNextLevelNodes >= BQ_CAPACITY so reset to BQ_CAPACITY
      unsigned int gq_idx = atomicAdd(numNextLevelNodes,1);
      nextLevelNodes[gq_idx] = w_nextLevelNodes[i][wqueue_idx];
    }
  }
  __syncthreads();


  // //update global numNextLevelNodes for other blocks to determine their start 
  if (threadIdx.x == 0){
    b_start = atomicAdd(numNextLevelNodes, b_numNextLevelNodes);
  }
  __syncthreads();
  for (unsigned int i = threadIdx.x; i < b_numNextLevelNodes; i += blockDim.x){
    nextLevelNodes[i+b_start] = b_nextLevelNodes[i];
  }
}

/******************************************************************************
 Functions
*******************************************************************************/

void cpu_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
  unsigned int *nodeVisited, unsigned int *currLevelNodes,
  unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
  unsigned int *numNextLevelNodes) {

  // Loop over all nodes in the curent level
  for(unsigned int idx = 0; idx < *numCurrLevelNodes; ++idx) {
    unsigned int node = currLevelNodes[idx];
    // Loop over all neighbors of the node113
    for(unsigned int nbrIdx = nodePtrs[node]; nbrIdx < nodePtrs[node + 1];
      ++nbrIdx) {
      unsigned int neighbor = nodeNeighbors[nbrIdx];
      // If the neighbor hasn't been visited yet
      if(!nodeVisited[neighbor]) {
        // Mark it and add it to the queue
        nodeVisited[neighbor] = 1;
        nextLevelNodes[*numNextLevelNodes] = neighbor;
        ++(*numNextLevelNodes);
      }
    }
  }

}

void gpu_global_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
  unsigned int *nodeVisited, unsigned int *currLevelNodes,
  unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
  unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_global_queuing_kernel <<< numBlocks , BLOCK_SIZE >>> (nodePtrs,
    nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
    numCurrLevelNodes, numNextLevelNodes);

}

void gpu_block_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
  unsigned int *nodeVisited, unsigned int *currLevelNodes,
  unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
  unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_block_queuing_kernel <<< numBlocks , BLOCK_SIZE >>> (nodePtrs,
    nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
    numCurrLevelNodes, numNextLevelNodes);

}

void gpu_warp_queuing(unsigned int *nodePtrs, unsigned int *nodeNeighbors,
  unsigned int *nodeVisited, unsigned int *currLevelNodes,
  unsigned int *nextLevelNodes, unsigned int *numCurrLevelNodes,
  unsigned int *numNextLevelNodes) {

  const unsigned int numBlocks = 45;
  gpu_warp_queuing_kernel <<< numBlocks , BLOCK_SIZE >>> (nodePtrs,
    nodeNeighbors, nodeVisited, currLevelNodes, nextLevelNodes,
    numCurrLevelNodes, numNextLevelNodes);

}

