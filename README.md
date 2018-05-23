# gpusim
GPU accelerated NORTA
Project for CS791: GPU programming. 
This project implements NORTA transformations for generating random variables with given correlation matrix.


# How to use program:
makefile is in build folder. <br />
Executable is called gpusim <br />
Command line arguments are: <br />
                  \<correlation matrix file path> \<distributions file path> d n seed \<output file name>

d is the number of random variables in random vector (also dimension of corr matrix ) <br />
n is the number of simulation replicates <br />
seed is the seed used for random number generator<br />
output file name is the file name of the simulation matrix file that will be generated at the end of program <br />

<br /> There is a space between each argument.

call goes like this: <br />
       gpusim ../test_corr_matrix_d=20501.txt ../betar20501.csv 20501 2000 154224

