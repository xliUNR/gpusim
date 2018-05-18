# gpusim
GPU accelerated NORTA
Project for CS791: GPU programming. 
This project implements NORTA transformations for generating random variables with given correlation matrix.


#How to use program:
makefile is in build folder.
Executable is called gpusim
Command line arguments are: 
                  <correlation matrix file path> <distributions file path> d n 

d is the number of random variables in random vector 
(also dimension of corr matrix )

n is the number of simulation replicates



call goes like this:
   gpusim ../test_corr_matrix_d=20501.txt ../betar20501.csv 20501 2000