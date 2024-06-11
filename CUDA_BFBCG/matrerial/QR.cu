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
#include "includes/helper_debug.h"
// helper function CUDA error checking and initialization
#include "includes/helper_cuda.h"  
#include "includes/helper_functions.h"
#include "includes/cusolver_utils.h"

#define CHECK(call){ \
    const cudaError_t cuda_ret = call; \
    if(cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
        printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
        exit(-1); \
    }\
}

/*
A = |1 2|
    |3 4|
*/

// Define the dense matrixA column way
float mtxA[] = {
    1.0, 3.0, 2.0, 4.0
};


int main(int argc, char** argv){
    const int row = 2; 
    const int clm = 2; 
    const int lda = row;

    float *mtxA_d = nullptr;

    // Scalar factors of the Householder reflectors, H
    // tau array is a vector that holds the scalar factors for the Householder transformations 
    // used in the QR factorization process.
    float *tau_d = nullptr;
    
    // Computing orthonomal set Q with the product of successive householder transformation
    // such that Q = H1 * H2 ... H_{n-1}
    float *mtxQ_d = nullptr;

    int lwork = 0; // place holder
    float *work_d = nullptr; // Need work space for functions
    int *devInfo = nullptr; // Need for invoking functions later

    bool debug = true;

    //(1) Allocate memoery in device
    CHECK(cudaMalloc((void**)& mtxA_d, sizeof(float) * lda * clm));
    CHECK(cudaMalloc((void**)&tau_d, sizeof(float) * clm));

    //(2) Copy data to device
    CHECK(cudaMemcpy(mtxA_d, mtxA, sizeof(float) * lda * clm, cudaMemcpyHostToDevice));
    if(debug){
        printf("\n\n~~mtxA~~\n");
        print_mtx_clm_d(mtxA_d, row, clm);
    }

    //(3) Createing cusolver handler
    cusolverDnHandle_t cusolverHander = nullptr;
    checkCudaErrors(cusolverDnCreate(&cusolverHander));
    
    //(4) Calculate and assign workspace
    checkCudaErrors(cusolverDnSgeqrf_bufferSize(cusolverHander, row, clm, mtxA_d, lda, &lwork));
    CHECK(cudaMalloc((void**)&work_d, sizeof(float) * lwork)); 
    CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));

    //(5) Compute A = QR
    // Performa QR factorization
    checkCudaErrors(cusolverDnSgeqrf(cusolverHander, row, clm, mtxA_d, lda, tau_d, work_d, lwork, devInfo));
    
    // Compose Q with product of successive householder transformation Q = H1H2....H_{row-1}
    // 6th parameter, Number of elementary reflextions whose product defines the matrix Q 
    // it is typically the same number of colmn of matrix A. 
    checkCudaErrors(cusolverDnSormqr(cusolverHander, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, row, clm, clm, mtxA_d, lda, tau_d, mtxA_d, lda, work_d, lwork, devInfo));

    if(debug){
        printf("\n\n~~Q~~\n");
        print_mtx_clm_d(mtxA_d, row, clm);
    }



    //Free memory
    checkCudaErrors(cusolverDnDestroy(cusolverHander));
    checkCudaErrors(cudaFree(mtxA_d));
    checkCudaErrors(cudaFree(mtxQ_d));
    checkCudaErrors(cudaFree(tau_d));
    checkCudaErrors(cudaFree(work_d));
    checkCudaErrors(cudaFree(devInfo));
    





    return 0;
} // end of main