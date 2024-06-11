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


void inverseTest_Case1();
void inverseTest_Case2();
void inverseTest_Case3();
void inverseTest_Case4();
void inverseTest_Case5();

int main(int arg, char** argv)
{
    printf("\n\nHello World from inverseTest.cu\n\n");

    inverseTest_Case1();
    inverseTest_Case2();
    inverseTest_Case3();
    inverseTest_Case4();
    inverseTest_Case5();

    return 0;
} // end of main


void inverseTest_Case1()
{

}


void inverseTest_Case2()
{
    
}


void inverseTest_Case3()
{
    
}


void inverseTest_Case4()
{
    
}


void inverseTest_Case5()
{
    
}