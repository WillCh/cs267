# on Hopper, we will benchmark you against Cray LibSci, the default vendor-tuned BLAS.
# The Cray compiler wrappers handle all the linking. If you wish to compare with other BLAS implementations, 
# check the NERSC documentation.
# This makefile is intended for the GNU C compiler. 
# On Hopper, the Portland compilers are default, so you must instruct the Cray compiler wrappers to switch to GNU:
# type "module swap PrgEnv-pgi PrgEnv-gnu"

CC = cc 
OPT = -O3 -ffast-math
CFLAGS = -Wall -std=gnu99 $(OPT) -msse2 
LDFLAGS = -Wall
# librt is needed for clock_gettime
LDLIBS = -lrt  

#change here for clean
targets = benchmark-naive benchmark-blocked benchmark-blas  benchmark-sse-block
#benchmark-sse benchmark-tran-sse benchmark-transposed benchmark-tran-sse-block benchmark-tran-block benchmark-sse-order benchmark-sse-block
objects = benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-sse-block.o dgemm-blas.o  benchmark-sse-block.o dgemm-sse-block.o
#dgemm-sse-order.o dgemm-tran-block.o dgemm-tran-sse.o dgemm-transposed.o benchmark-sse.o benchmark-tran-sse.o benchmark-transposed.o benchmark-tran-sse-block.o  benchmark-tran-block.o  benchmark-sse-order benchmark-sse-block.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o 
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o dgemm-blocked.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o
	$(CC) -o $@ $^ $(LDLIBS)
#benchmark-sse : benchmark.o dgemm-sse.o
#	$(CC) -o $@ $^ $(LDLIBS)
#benchmark-tran-sse : benchmark.o dgemm-tran-sse.o
#	$(CC) -o $@ $^ $(LDLIBS)
#benchmark-transposed : benchmark.o dgemm-transposed.o
#	$(CC) -o $@ $^ $(LDLIBS)
#benchmark-tran-sse-block : benchmark.o dgemm-tran-sse-block.o
#	$(CC) -o $@ $^ $(LDLIBS)
benchmark-tran-block : benchmark.o dgemm-tran-block.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-sse-order : benchmark.o dgemm-sse-order.o
	$(CC) -o $@ $^ $(LDLIBS)
benchmark-sse-block : benchmark.o dgemm-sse-block.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
