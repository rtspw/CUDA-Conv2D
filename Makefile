C=nvcc

OPT?=-O3
CFLAGS= --std=c++11 -g $(OPT) 

conv: convolution.cu
	$(C) $^ $(CFLAGS) -o $@ -DNx=224 -DNy=224 -DKx=3 -DKy=3 -DNi=64 -DNn=64 -DTii=32 -DTi=16 -DTnn=32 -DTn=16 -DTx=7 -DTy=7
