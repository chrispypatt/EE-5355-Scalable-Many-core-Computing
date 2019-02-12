# Lab 3: Matrix Multiplication with Thread Coarsening and Register Tiling

The purpose of this lab is to practice thread coarsening and register tiling op- timization techniques using matrix-matrix multiplication as an example.

## Running code:
```
make
./sgemm−tiled               # Uses the default matrix sizes
./sgemm−tiled <m>           # Uses square m x m matrices
./sgemm−tiled <m> <k> <n>   # Uses (m x k) and (k x n) input matrices
```

