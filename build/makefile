CC=/usr/local/cuda-9.0/bin/nvcc
INCLUDES=-I../include
O_FILES=gpunorta.o
LIBS = -lcusolver -lcudart
LIBDIRS= -L/usr/local/cuda-9.0/lib64
all: $(O_FILES)
	 $(CC) -o test1 $(O_FILES) $(LIBDIRS) $(LIBS)
	 
	
gpunorta.o: ../src/gpunorta.cu
	 $(CC) -c ../src/gpunorta.cu -o gpunorta.o $(LIBDIRS) $(INCLUDES) $(LIBS)
	 

clean:
	rm -f *.o
	rm -f *~

		
