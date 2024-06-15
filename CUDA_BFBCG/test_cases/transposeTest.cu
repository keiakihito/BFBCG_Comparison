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
#include "../includes/CSRMatrix.h"


#define CHECK(call){ \
    const cudaError_t cuda_ret = call; \
    if(cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
        printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
        exit(-1); \
    }\
}


void transposeTest_Case1();
void transposeTest_Case2();
void transposeTest_Case3();
void transposeTest_Case4();
void transposeTest_Case5();

int main(int arg, char** argv)
{
    
    printf("\n\n= = = =tranposeTest.cu= = = = \n\n");
    
    // printf("\n\n🔍🔍🔍 Test Case 1 🔍🔍🔍\n\n");
    // tranposeTest_Case1();

    // printf("\n\n🔍🔍🔍 Test Case 2 🔍🔍🔍\n\n");
    // tranposeTest_Case2();

    // printf("\n\n🔍🔍🔍 Test Case 3 🔍🔍🔍\n\n");
    // tranposeTest_Case3();

    // printf("\n\n🔍🔍🔍 Test Case 4 🔍🔍🔍\n\n");
    // tranposeTest_Case4();

    // printf("\n\n🔍🔍🔍 Test Case 5 🔍🔍🔍\n\n");
    // tranposeTest_Case5();

    printf("\n\n= = = = end of tranposeTest = = = =\n\n");

    return 0;
} // end of main




void tranposeTest_Case1()
{

} // end of tranposeTest_Case1




void tranposeTest_Case2()
{

} // end of tranposeTest_Case2




void tranposeTest_Case3()
{

} // end of tranposeTest_Case3




void tranposeTest_Case4()
{

} // end of tranposeTest_Case4




void tranposeTest_Case5()
{

} // end of tranposeTest_Case5