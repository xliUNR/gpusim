CC=/usr/local/cuda-9.0/bin/nvcc
INCLUDES=-I../include
O_FILES=gpunorta.o cudaFuncs.o
LIBS = -lcusolver -lcudart -lcurand -lcublas
LIBDIRS= -L/usr/local/cuda-9.0/lib64
CC_FLAGS= -Xcompiler -fopenmp -g -G -std=c++11 
all: $(O_FILES)
	 $(CC) -o test1 $(O_FILES) $(LIBDIRS) $(LIBS) $(CC_FLAGS) 
	 
	
gpunorta.o: ../src/gpunorta.cu
	 $(CC) -c ../src/gpunorta.cu -o gpunorta.o $(LIBDIRS) $(INCLUDES) $(LIBS) -std=c++11  --expt-relaxed-constexpr -w

	 
cudaFuncs.o: ../src/cudaFuncs.cu
	 $(CC) -c ../src/cudaFuncs.cu -o cudaFuncs.o $(LIBDIRS) $(INCLUDES) $(LIBS) -std=c++11 --expt-relaxed-constexpr -w

clean:
	rm -f *.o
	rm -f *~

		
