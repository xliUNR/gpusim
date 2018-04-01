///////////////////////////////////////////////////////////////////////////////
///////////////   This is the GPU version of NORTA   //////////////////////////
///////////////////// Written by Eric Li //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////



/////////////////////////////  Includes  //////////////////////////////////////
#include <cstdlib>
#include <stdio>








///////////////////////////////////////////////////////////////////////////////
/////////////////////////////////    Main   ///////////////////////////////////
int main( int argc, char const *argv[])
{
 //initialize variables
   //initiliaze arrays for holding input data
   float *r20, *r200;
   int r20Dim, r200Dim;
   r20Dim = 20;
   r200Dim = 200;
   r20 = (float*)malloc(20*20*sizeof(float));
   r200 = (float*)malloc(200*200*sizeof(float));;
	   
   //Section for reading in arrays from file
   FILE * fp;
   fp = fopen("test_corr_matrix_d=20.txt", "r");
   if(fp)
      {
        for(int i = 0; i < 20*20; i++)
          {
           r20[i] = fscanf(fp, "%f");
          } 
      }
    else
      {
        printf("\n ERROR OPENING FILE");
      }

   fclose(fp);
   fp = fopen("test_corr_matrix_d=200.txt", "r"); 
   if(fp)
      {
        for(int i = 0; i < 200 * 200; i++)
           {
              r200[i] = fscanf(fp, "%f");    
      }

   //test input read by printing results
   for(int i = 0; i < 20; i++ ){
    for(int j = 0; j <20; j++ )
      {
        printf("%f", r20[i*20+j]);
      } 
      printf("\n");
   }   

    for(int i = 0; i < 200; i++ ){
      for(int j = 0; j <200; j++ )
        {
          printf("%f", r200[i*20+j]);
        } 
        printf("\n");
   }  
      
   //free memory
   free(r20);
   free(r200);

   
}
