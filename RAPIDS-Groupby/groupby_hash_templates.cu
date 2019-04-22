#include <cstddef>

#include "limits.cuh"

// assume column major here
template <typename T> __host__ __device__
bool keyEqualCM(T* key_columns, size_t idx1, size_t idx2, size_t num_key_rows, size_t num_key_columns)
{
  for (size_t i=0; i < num_key_columns; ++i) {
    if (key_columns[i*num_key_rows+idx1] != key_columns[i*num_key_rows+idx2])
      return false;
  }
  return true;
}

// assume row major here
template <typename T> __host__ __device__
bool keyEqualRM(T* key_columns, size_t idx1, size_t idx2, size_t num_key_rows, size_t num_key_columns)
{
  for (size_t i=0; i < num_key_columns; ++i) {
    if (key_columns[i+num_key_rows*idx1] != key_columns[i+num_key_rows*idx2])
      return false;
  }
  return true;
}

// placeholder
__host__ __device__
size_t HashKey(size_t idx) {
  return 0;
}

template <typename Tval> __device__
void updateEntry(Tval* value_columns,
		 size_t num_val_rows,
		 reductionType* ops,
		 size_t num_ops,
		 size_t idx,
		 size_t hashPos,
		 Tval* hash_results,
		 int* countPtr,
		 size_t len_hash_table)
{
  // update count
  atomicAdd(countPtr, 1);
  // update each item
  for (size_t i = 0; i < num_ops; ++i) {
    Tval value = value_columns[i * num_val_rows + idx];
    size_t val_idx = i * len_hash_table + hashPos;
    switch(ops[i]) {
    case rmin:
      atomicMin(&(hash_results[val_idx]), value);
      break;
    case rmax:
      atomicMax(&(hash_results[val_idx]), value);
      break;
    case rcount:
      atomicAdd(&(hash_results[val_idx]), 1);
      break;
    case rmean: // fall-thru
    case rsum:
      atomicAdd(&(hash_results[val_idx]), value);
      break;
    }
  }
}


template <typename Tkey, typename Tval> __global__
void fillTable(Tkey* key_columns,
	       size_t num_key_rows,
	       size_t num_key_cols,
	       Tval* value_columns,
	       size_t num_val_rows,
	       size_t num_val_cols,
	       int* hash_key_idx,
	       int* hash_count,
	       Tval* hash_results,
	       size_t len_hash_table,
	       reductionType* ops,
	       size_t num_ops)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t offset = gridDim.x * blockDim.x;
  for (size_t i = idx; i < num_key_rows; i += offset) {
    // try inserting, assume there is enough space
    size_t curPos = HashKey(i) % len_hash_table;
    unsigned int collisionCount = 0;
    bool isInserted = false;
    while (!isInserted) {
      // first read the value out 
      ssize_t old = hash_key_idx[curPos];
      // if it is -1, try update, else don't
      if (old == -1) 
	old = atomicCAS(&(hash_key_idx[curPos]), -1, i);
      // now old contains either i or a new address, if it is a new address meaning other thread claimed it
      // note: old should not contain -1 now, safe to cast to size_t
      if ((size_t)old != i) {
	if (!keyEqualCM<Tkey>(key_columns, (size_t)old, i, num_key_rows, num_key_cols)) {
	  // collision
	  curPos = (curPos + 1) % len_hash_table; // linear probing
	  if (++collisionCount == len_hash_table)
	    break; // break the loop if it looped over the hash table and still failed
	  continue;
	}
      }
      // now it is safe to update the entry
      isInserted = true;
      updateEntry<Tval>(value_columns, num_val_rows, ops, num_ops, i, curPos, hash_results, &(hash_count[curPos]), len_hash_table);
    }
    if (!isInserted) {
      // Do sth in the case of overflowing hash table
    }
  }
}

template <typename Tval> __global__
void initializeVariable(int* hash_key_idx,
			int* hash_count,
			Tval* hash_results,
			size_t len_hash_table,
			reductionType* ops,
			size_t num_ops)
{
  // each thread responsible for one entry (with thread coarsening)
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t offset = gridDim.x * blockDim.x;
  for (size_t i = idx; i < len_hash_table; i += offset) {
    hash_key_idx[i] = -1;
    hash_count[i] = 0;
    for (size_t j = 0; j < num_ops; ++j) {
      // replace following with specialized limit template in the future
      if (ops[i] == rmin) {
	hash_results[j * len_hash_table + i] = cuda_custom::limits<Tval>::max();
      } else if (ops[i] == rmax) {
        hash_results[j * len_hash_table + i] = cuda_custom::limits<Tval>::lowest();
      } else {
	hash_results[j * len_hash_table + i] = 0;
      }
    }
  }
}

template <typename Tval> __global__
void copyUnique(
      int *hashTable_idxs_d, //where key resides in hash vector
      int *hash_key_idx_d, //where key resides in original key matrix 
      Tval* key_columns_d, 
      Tval* output_key_columns_d, 
      int num_output_rows, 
      int num_key_columns.
      int num_key_rows)
{
  idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < num_output_rows){
    for (int i = 0; i < num_key_columns, i++){//each column of key matrix
      output_key_columns_d[idx+num_output_rows*i] = key_columns_d[hash_key_idx_d[hashTable_idxs_d[idx]]+num_key_rows*i];//copy original key entry to output
    }
    idx += gridDim.x*blockDim.x;//increment idx by thread space
  }
}

template <typename Tval> __global__
void copyValues(
      int *hashTable_idxs_d, 
      Tval* hash_results_d,
      int *hash_count_d,
      Tval* value_columns_d, 
      Tval* output_value_columns_d, 
      int num_output_rows, 
      int num_value_columns.
      int num_value_rows,
      reductionType* ops,
      size_t num_ops,
      size_t len_hash_table,
    )
{
  idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < num_output_rows){
    for (size_t i = 0; i < num_ops; ++i) {
      size_t val_idx = i * len_hash_table + hashTable_idxs_d[idx];
      switch(ops[i]) {
      case rmin:
        output_value_columns_d[idx+num_output_rows*i] = hash_results_d[val_idx];//copy result to output
        break;
      case rmax:
        output_value_columns_d[idx+num_output_rows*i] = hash_results_d[val_idx];//copy result to output
      break;
      case rcount:
        output_value_columns_d[idx+num_output_rows*i] = hash_results_d[val_idx];//copy result to output
      break;
      case rmean: 
        output_value_columns_d[idx+num_output_rows*i] = hash_results_d[val_idx]/hash_count_d[val_idx];//copy result to output
        break;
      case rsum:
        output_value_columns_d[idx+num_output_rows*i] = hash_results_d[val_idx];//copy result to output
        break;
      }
    }

    idx += gridDim.x*blockDim.x;//increment idx by thread space
  }
}

struct is_not_neg_1
{
  __host__ __device__
  bool operator()(const int x)
  {
    return x != -1s;
  }
};
