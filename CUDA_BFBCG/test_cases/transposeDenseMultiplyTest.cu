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


void transposeDenseMultiplyTest_Case1();
void transposeDenseMultiplyTest_Case2();
void transposeDenseMultiplyTest_Case3();
void transposeDenseMultiply_Case4();
void transposeDenseMultiply_Case5();

int main(int arg, char** argv)
{
    
    printf("\n\n= = = =tranposeTest.cu= = = = \n\n");
    
    printf("\n\n🔍🔍🔍 Test Case 1 🔍🔍🔍\n\n");
    transposeDenseMultiplyTest_Case1();

    // printf("\n\n🔍🔍🔍 Test Case 2 🔍🔍🔍\n\n");
    // transposeDenseMultiplyTest_Case2();

    // printf("\n\n🔍🔍🔍 Test Case 3 🔍🔍🔍\n\n");
    // transposeDenseMultiplyTest_Case3();

    // printf("\n\n🔍🔍🔍 Test Case 4 🔍🔍🔍\n\n");
    // transposeDenseMultiplyTest_Case4();

    // printf("\n\n🔍🔍🔍 Test Case 5 🔍🔍🔍\n\n");
    // transposeDenseMultiplyTest_Case5();

    printf("\n\n= = = = end of tranposeTest = = = =\n\n");

    return 0;
} // end of main




void transposeDenseMultiplyTest_Case1()
{
    float mtxA[] = {4.0, 1.0, 1.0, 
                    1.0, 3.0, 1.0};
    float* mtxA_d = NULL;
    float* mtxAT_d = NULL;

    int numOfRow = 3;
    int numOfCol = 2;

    bool debug = true;

    //Crete handler
    cusolverDnHandle_t cusolverHandler;
    checkCudaErrors(cusolverDnCreate(&cusolverHandler));


    //(1) Allocate device memeory
    CHECK(cudaMalloc((void**)&mtxA_d, numOfRow * numOfCol * sizeof(float)));
    CHECK(cudaMalloc((void**)&mtxAT_d, numOfRow * numOfCol * sizeof(float)));

    //(2) Copy data to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, numOfRow * numOfCol * sizeof(float), cudaMemcpyHostToDevice));
    
    if(debug){
        printf("\n\n = = = Before transpose operation = = =\n");
        printf("\n\n~~mtxA_d~~\n\n");
        print_mtx_clm_d(mtxA_d, numOfRow, numOfCol);
    }

    //(3) Set up cublas handler
    cublasHandle_t cublasHandler;
    checkCudaErrors(cublasCreate(&cublasHandler));

    const float alpha = 1.0f;
    const float beta = 0.0;

    //(4) Transpose operatoin
    checkCudaErrors(cublasSgeam(cublasHandler, CUBLAS_OP_N, CUBLAS_OP_T, numOfRow, numOfCol, &alpha, mtxA_d, numOfCol, &beta, mtxA_d, numOfRow, mtxAT_d, numOfRow));

    if(debug){
        printf("\n\n = = = After transpose operation = = =\n");
        printf("\n\n~~mtxA_d~~\n\n");
        print_mtx_clm_d(mtxA_d, numOfRow, numOfCol);
        printf("\n\n~~mtxAT_d~~\n\n");
        print_mtx_clm_d(mtxAT_d, numOfCol, numOfRow);
    }

    checkCudaErrors(cublasDestroy(cublasHandler));
    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxAT_d));
} // end of tranposeTest_Case1




void transposeDenseMultiplyTest_Case2()
{

} // end of tranposeTest_Case2




void transposeDenseMultiplyTest_Case3()
{

} // end of tranposeTest_Case3




void tranposeDenseMultiplyTest_Case4()
{

} // end of tranposeTest_Case4




void transposeDenseMultiplyTest_Case5()
{

} // end of tranposeTest_Case5