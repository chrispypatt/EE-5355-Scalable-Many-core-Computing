
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include -arch=sm_20
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = s2g
OBJ	        = main.o

default: $(EXE)

main.o: main.cu kernel.cu support.cu support.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
