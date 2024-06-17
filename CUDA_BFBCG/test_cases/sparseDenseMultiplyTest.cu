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


void sparseDenseMultiplyTest_Case1();
void sparseDenseMultiplyTest_Case2();
void sparseDenseMultiplyTest_Case3();
void sparseDenseMultiplyTest_Case4();
void sparseDenseMultiplyTest_Case5();

int main(int arg, char** argv)
{
    
    printf("\n\n= = = =sparseDenseMultiplyTest.cu= = = = \n\n");
    
    printf("\n\n🔍🔍🔍 Test Case 1 🔍🔍🔍\n\n");
    sparseDenseMultiplyTest_Case1();

    // printf("\n\n🔍🔍🔍 Test Case 2 🔍🔍🔍\n\n");
    // sparseDenseMultiplyTest_Case2();

    // printf("\n\n🔍🔍🔍 Test Case 3 🔍🔍🔍\n\n");
    // sparseDenseMultiplyTest_Case3();

    // printf("\n\n🔍🔍🔍 Test Case 4 🔍🔍🔍\n\n");
    // sparseDenseMultiplyTest_Case4();

    // printf("\n\n🔍🔍🔍 Test Case 5 🔍🔍🔍\n\n");
    // sparseDenseMultiplyTest_Case5();

    printf("\n\n= = = = end of sparseDenseMultiplyTest = = = =\n\n");

    return 0;
} // end of main




void sparseDenseMultiplyTest_Case1()
{
    // Number of matrix A rows and columns
    const int N = 5;

    // Define the dense matrixB column major
    float dnsMtxB_h[] = {
        0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 1.0,
        1.1, 1.2, 1.3, 1.4, 1.5
    };

    int numOfRowsB = 5;
    int numOfClmsB = 3;

    float* dnsMtxB_d = NULL;
    float* dnsMtxC_d = NULL;

    bool debug = true;



    //Generate sparse Identity matrix
    CSRMatrix csrMtxI_h = generateSparseIdentityMatrixCSR(N);


    if(debug){
        
        //Sparse matrix information to check
        printf("\n\n~~mtxI sparse~~\n\n");
        print_CSRMtx(csrMtxI_h);
        
        //Converst CSR to dense matrix to check
        float* dnsMtx = csrToDense(csrMtxI_h);
        float *mtxI_d = NULL;
        CHECK(cudaMalloc((void**)&mtxI_d, N * N * sizeof(float)));
        CHECK(cudaMemcpy(mtxI_d, dnsMtx, N * N * sizeof(float), cudaMemcpyHostToDevice));
        
        printf("\n\n~~mtxI dense~~\n\n");
        print_mtx_clm_d(mtxI_d, N, N);
        
        CHECK(cudaFree(mtxI_d));
        free(dnsMtx);
    }


    //(1) Allocate memeory 
    CHECK(cudaMalloc((void**)&dnsMtxB_d, numOfRowsB * numOfClmsB * sizeof(float)));
    CHECK(cudaMalloc((void**)&dnsMtxC_d, N * numOfClmsB * sizeof(float)));

    //(2) Copy values from host to device
    CHECK(cudaMemcpy(dnsMtxB_d, dnsMtxB_h, numOfRowsB * numOfClmsB * sizeof(float), cudaMemcpyHostToDevice));

    if(debug){
        //Sparse matrix information to check
        printf("\n\n~~dense matrix B~~\n\n");
        print_mtx_clm_d(dnsMtxB_d, numOfRowsB, numOfClmsB);
    }

    //Call sparseMultiplySense function
    multiply_Src_Den_mtx(csrMtxI_h, dnsMtxB_d, numOfClmsB, dnsMtxC_d);

    //Get dense matrix as a result
    printf("\n\n~~dense matrix C after multiplication~~\n\n");
    print_mtx_clm_d(dnsMtxC_d, N, numOfClmsB);


    //Free memory
    CHECK(cudaFree(dnsMtxB_d));
    CHECK(cudaFree(dnsMtxC_d));
    freeCSRMatrix(csrMtxI_h);



} // end of sparseDenseMultiplyTest_Case1




void sparseDenseMultiplyTest_Case2()
{

} // end of sparseDenseMultiplyTest_Case2




void sparseDenseMultiplyTest_Case3()
{

} // end of sparseDenseMultiplyTest_Case1




void sparseDenseMultiplyTest_Case4()
{

} // end of sparseDenseMultiplyTest_Case1




void sparseDenseMultiplyTest_Case5()
{

} // end of sparseDenseMultiplyTest_Case1