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
    
    printf("\n\n= = = =inverseTest.cu= = = = \n\n");
    printf("\n\n🔍🔍🔍 Test Case 1 🔍🔍🔍\n\n");
    inverseTest_Case1();

    // inverseTest_Case2();
    // inverseTest_Case3();
    // inverseTest_Case4();
    // inverseTest_Case5();

    printf("\n\n= = = = end of invereTest = = = =\n\n");

    return 0;
} // end of main


void inverseTest_Case1()
{

    /*
    Compute inverse, X with LU factorization
    A = LU
    LU * X = I
    L *(UX) = L * Y = I

    Solve X
        UX = Y

    */

    /*
    mtxA =  |4 1|
            |1 3|

    Answer
    mtxA^(-1) = | 3/11 -1/11|
                |-1/11  4/11|

    or

    mtxA^(-1) = |0.2727  -0.091|
                |-0.091  0.3636|
    */

    //Defince the dense matrix A column major
    float mtxA[] = {4.0, 1.0, 1.0, 3.0};
    float* mtxA_d = NULL;
    float* mtxA_inv_d = NULL;

    const int N = 2;

    bool debug = true;

    //Crete handler
    cusolverDnHandle_t cusolverHandler;
    checkCudaErrors(cusolverDnCreate(&cusolverHandler));


    //(1) Allocate device memeory
    CHECK(cudaMalloc((void**)&mtxA_d, N * N * sizeof(float)));
    CHECK(cudaMalloc((void**)&mtxA_inv_d, N * N * sizeof(float)));

    //(2) Copy data to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, N * N * sizeof(float), cudaMemcpyHostToDevice));
    
    //(3) Perform inverse operation
    inverse_Den_Mtx(cusolverHandler, mtxA_d, mtxA_inv_d, N);

    //(4) Check the result
    if(debug){
        printf("\n\nCompute inverse\n");
        printf("\n\n~~mtxA inverse~~\n\n");
        print_mtx_clm_d(mtxA_inv_d, N, N);
    }

    //(5) Free memory
    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxA_inv_d));

} // end of inverseTest_Case1


void inverseTest_Case2()
{
    
}// end of inverseTest_Case2


void inverseTest_Case3()
{
    
}// end of inverseTest_Case3


void inverseTest_Case4()
{
    
}// end of inverseTest_Case4


void inverseTest_Case5()
{
    
}// end of inverseTest_Case5