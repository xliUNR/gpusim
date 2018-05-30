CC=/usr/local/cuda-9.1/bin/nvcc
INCLUDES=-I../include
O_FILES=gpunorta_cli.o cudaFuncs.o
LIBS = -lcusolver -lcudart -lcurand -lcublas
LIBDIRS= -L/usr/local/cuda-9.1/lib64
CC_FLAGS= -Xcompiler -fopenmp -g -G -std=c++11 
all: $(O_FILES)
	 $(CC) -o gpusim $(O_FILES) $(LIBDIRS) $(LIBS) $(CC_FLAGS) 
	 
	
gpunorta_cli.o: ../src/gpunorta_cli.cu
	 $(CC) -c ../src/gpunorta_cli.cu -o gpunorta_cli.o $(LIBDIRS) $(INCLUDES) $(LIBS) -std=c++11  --expt-relaxed-constexpr -w

	 
cudaFuncs.o: ../src/cudaFuncs.cu
	 $(CC) -c ../src/cudaFuncs.cu -o cudaFuncs.o $(LIBDIRS) $(INCLUDES) $(LIBS) -std=c++11 --expt-relaxed-constexpr -w

clean:
	rm -f *.o
	rm -f *~

		
