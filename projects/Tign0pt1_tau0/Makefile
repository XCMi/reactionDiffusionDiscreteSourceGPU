#
# Copyright © Cambridge Numerical Solutions Ltd 2013
#
.SUFFIXES: .cu .cpp

all: ghost

TARGET = ghost

SRCDIR = src

SOURCE = ghost.cu

# Find picture and include dependencies from a tex file
%.d: %.cu
	nvcc $< --compiler-bindir=/usr/local -M | sed 's,\($*\)\.o[ :]*,\1 $@ : ,g' > $@

NVCC:= /usr/local/cuda-8.0/bin/nvcc
CUDA_DIR:=$(dir $(NVCC))../
CUDA_INC:=$(CUDA_DIR)/include
ARCH:=$(shell uname -m)

ifeq ($(ARCH),x86_64)
CUDA_LIB:=$(CUDA_DIR)/lib64
else
CUDA_LIB:=$(CUDA_DIR)/lib
endif

$(TARGET): $(SOURCE)
	$(NVCC) $< -o $@ --compiler-bindir=/usr/bin/ -Xptxas -v -arch=compute_35 -code=sm_35 -L/usr/local/cuda-8.0/lib64 -L/usr/local/cuda-8.0/samples/common/lib/linux/ -I/usr/local/cuda-8.0/samples/common/inc -I../../src -I. -lrt -lcuda -lboost_thread -lboost_system -lglut -lpng -lconfig++
