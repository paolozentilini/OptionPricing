LIBS= -lcuda -lm -L/usr/local/cuda/lib/ -lcudart
INCLUDE=-I/home/vicini/NVIDIA_GPU_Computing_SDK/CUDALibraries/common/inc/ -I/home/vicini/NVIDIA_GPU_Computing_SDK/shared/inc/ -I../../common/ -I/usr/local/cuda/include/
OBJS=

NVCC=nvcc

ECHO=/bin/echo

default: main_gpu.x


%.x: %.cu
	$(NVCC) $(INCLUDE) $(OBJS) $(LIBS) $< -o $@

clean:
	rm -f *.x *.o

esegui:
	./main_gpu.x
