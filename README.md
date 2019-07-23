# gpusim
A graphical processing unit accelerated NOrmal To Anything algorithm for high dimensional multivariate simulation

Project for CS791: GPU programming. <br />
This project implements NORTA transformations for generating random variables with given correlation matrix. <br />
Presented at ITNG 2019. Won best student paper award. </br>

Link to publication: https://link.springer.com/chapter/10.1007/978-3-030-14070-0_46

# How to use program:
makefile is in build folder. <br />
Executable is called gpusim <br />
Command line arguments are: <br />
                  \<correlation matrix file path> \<distributions file path> d n seed \<output file name> runtype inverseflag

d is the number of random variables in random vector (also dimension of corr matrix ) <br />
n is the number of simulation replicates <br />
seed is the seed used for random number generator<br />
output file name is the file name of the simulation matrix file that will be generated at the end of program <br />

runtype is an integer. 0 for sequential run, 1 for parallel GPU. <br />
inverseflag is an integer. 0 to not run inverse transform, 1 to run inverse transform. <br />

<br /> There is a space between each argument.

call goes like this: <br />
       gpusim ../test_corr_matrix_d=20501.txt ../betar20501.csv 20501 2000 154224 simOutputFile 0 0

