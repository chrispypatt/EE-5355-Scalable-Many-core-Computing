
NVCC        = nvcc 
NVCC_FLAGS  = -I/usr/local/cuda/include 
ifdef dbg
	NVCC_FLAGS  += 	-g -G
else
	NVCC_FLAGS  += -O3
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = stencil
OBJ	        = main.o support.o


default: $(EXE)

main.o: main.cu kernel.cu support.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
