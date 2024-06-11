// includes, system
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

/*Using updated (v2) interfaces to cublas*/
#include<cublas_v2.h>
#include<cuda_runtime.h>
#include<cusparse.h>
#include <cusolverDn.h>
#include<sys/time.h>


//Utilities
#include "../includes/helper_debug.h"
// helper function CUDA error checking and initialization
#include "../includes/helper_cuda.h"  
#include "../includes/helper_functions.h"
#include "../includes/cusolver_utils.h"


#define CHECK(call){ \
    const cudaError_t cuda_ret = call; \
    if(cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
        printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
        exit(-1); \
    }\
}


void stopTest_Case1();
void stopTest_Case2();
void stopTest_Case3();
void stopTest_Case4();
void stopTest_Case5();

int main(int arg, char** argv)
{
    printf("\n\nHello World from stopTest.cu\n\n");

    stopTest_Case1();
    stopTest_Case2();
    stopTest_Case3();
    stopTest_Case4();
    stopTest_Case5();
    return 0;
} // end of main


void stopTest_Case1()
{

}


void stopTest_Case2()
{
    
}


void stopTest_Case3()
{
    
}


void stopTest_Case4()
{
    
}


void stopTest_Case5()
{
    
}