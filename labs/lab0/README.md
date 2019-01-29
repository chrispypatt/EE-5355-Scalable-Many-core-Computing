# Lab 0:

The purpose of this lab is to check your environment settings and to make sure you can compile and run CUDA programs on the environment you will be usingthroughout the course.  In this lab, you will:
- Obtain a sample assignment package and walk through the directory struc-ture
- Set up the environment for executing the assignments
- Test the environment with a simple program that just queries the GPUdevice

## Running code:
```
make
./device−query
```

##Output:

The program was run on the UMN GPU lab machine #06. The result was the following output:

```
There are 3 devices supporting CUDA

Device 0: "GeForce GTX 1080"
  Major revision number:                         6
  Minor revision number:                         1
  Total amount of global memory:                 4219011072 bytes
  Number of multiprocessors:                     20
  Number of cores:                               160
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per block:           1024
  Maximum sizes of each dimension of a block:    1024 x 1024 x 64
  Maximum sizes of each dimension of a grid:     2147483647 x 65535 x 65535
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Clock rate:                                    1.73 GHz
  Concurrent copy and execution:                 Yes

Device 1: "Quadro 2000"
  Major revision number:                         2
  Minor revision number:                         1
  Total amount of global memory:                 1011089408 bytes
  Number of multiprocessors:                     4
  Number of cores:                               32
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 32768
  Warp size:                                     32
  Maximum number of threads per block:           1024
  Maximum sizes of each dimension of a block:    1024 x 1024 x 64
  Maximum sizes of each dimension of a grid:     65535 x 65535 x 65535
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Clock rate:                                    1.25 GHz
  Concurrent copy and execution:                 Yes

Device 2: "GeForce GTX 480"
  Major revision number:                         2
  Minor revision number:                         0
  Total amount of global memory:                 1546125312 bytes
  Number of multiprocessors:                     15
  Number of cores:                               120
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 32768
  Warp size:                                     32
  Maximum number of threads per block:           1024
  Maximum sizes of each dimension of a block:    1024 x 1024 x 64
  Maximum sizes of each dimension of a grid:     65535 x 65535 x 65535
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Clock rate:                                    1.40 GHz
  Concurrent copy and execution:                 Yes

TEST PASSED
```
