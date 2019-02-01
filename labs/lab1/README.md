# Lab 1: Scatter-to-Gather Transformation

The purpose of this lab is to understand scatter-to-gather transformation in thecontext of a simple computation example.

Scatter patterns on the GPU assign one thread to each input element, perform and operation, and store the result in an output element. The issue arises of many input threads trying to update the same output element at the same time which can lead to race conditions. Therefore, Scatter implementations typically require atomic update operations to ensure there is no contention.

Gather, on the other hand, assigns one thread to each output element which iterates through each input and performs the calculation. There is only one thread performing an update to any given output so we don't require atomic operations but we lose out on opportunities to reuse calculations. In turn, this can cause performance to not be as expected.

## Running code:
```
make
./s2g <m>		# Mode : m,  Input :  4 ,096 ,  Output :  4 ,096
./s2g <m> <M>		# Mode : m,  Input :M,  Output :M
./s2g <m> <M> <N>	# Mode : m,  Input :M,  Output :N
```

- Mode 1 executes the CPU scatter version
- Mode 2 executes the CPU gather version
- Mode 3 executes the GPU scatter version
- Mode 4 executes the GPU gather version


## Mode Runtimes:
Modes @ input size = 4096 | output size = 4096:
- 1 (CPU Scatter): 0.115627 s
- 2 (CPU Gather): 0.115492 s
- 3 (GPU Scatter): 0.004376 s
- 4 (GPU Gather): 0.001128 s

Modes @ input size = 1000000 | output size = 4096:
- 1 (CPU Scatter): 28.251139 s
- 2 (CPU Gather):  28.450750 s
- 3 (GPU Scatter): 0.308622 s
- 4 (GPU Gather):  0.274912 s

Modes @ input size = 4096 | output size = 1000000:
- 1 (CPU Scatter): 28.368822 s
- 2 (CPU Gather): 28.193960 s
- 3 (GPU Scatter): 0.283202 s
- 4 (GPU Gather): 0.078519 s

Modes @ input size = 100000 | output size = 100000: (*note: attempted to do 1milx1mil
but was going to take too long on CPU)
- 1 (CPU Scatter): 69.048180 s
- 2 (CPU Gather): 69.009079 s
- 3 (GPU Scatter): 0.445995 s
- 4 (GPU Gather): 0.257394 s

Modes @ input size = 256 | output size = 256:
- 1 (CPU Scatter): 0.000450  s
- 2 (CPU Gather): 0.000458 s
- 3 (GPU Scatter): 0.000115 s
- 4 (GPU Gather): 0.000136 s