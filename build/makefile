CC=/usr/local/cuda-9.0/bin/nvcc
INCLUDES=-I../include
O_FILES=gpunorta.o

all: $(O_FILES)
	 $(CC) -o test1 $(O_FILES)

gpunorta.o: ../src/gpunorta.cu
	 $(CC) -c ../src/gpunorta.cu -o gpunorta.o $(INCLUDES)


clean:
	rm -f *.o
	rm -f *~

		
