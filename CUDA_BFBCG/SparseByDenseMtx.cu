// includes, system
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

/*Using updated (v2) interfaces to cublas*/
#include<cublas_v2.h>
#include<cuda_runtime.h>
#include<cusparse.h>
#include<sys/time.h>


//Utilities
#include "includes/helper_debug.h"
// helper function CUDA error checking and initialization
#include "includes/helper_cuda.h"  
#include "includes/helper_functions.h"

#define CHECK(call){ \
    const cudaError_t cuda_ret = call; \
    if(cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
        printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
        exit(-1); \
    }\
}

//Bigger size matrix
#define N 5 //


// Define the dense matrixB
float denseMtxB[] = {
    0.1, 0.6, 1.1,
    0.2, 0.7, 1.2,
    0.3, 0.8, 1.3,
    0.4, 0.9, 1.4,
    0.5, 1.0, 1.5
};







// Time tracker for each iteration
double myCPUTimer()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}




int main(int argc, char** argv)
{   
    // double startTime, endTime;
    int row[] = {0, 2, 5, 8, 11, 13};
    int col[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    float val[] = {10, 1, 1, 20, 1, 1, 30, 1, 1, 40, 1, 1, 50};

    //For sparse matrix A
    int numRows = 5;
    int numCols = 5;
    int nnz = 13;// Number of Non zero

    //For dense matrix B
    int numRows_B = 5;
    int numCols_B = 3;

    //(1) Allocate device memory
    int *row_d = NULL;
    int *col_d = NULL;
    float *val_d = NULL;

    float *dnsMtxB_d = NULL;
    float *dnsMtxAB_d = NULL;// Result

    CHECK(cudaMalloc((void**)&row_d, (numRows+1) * sizeof(int)));
    CHECK(cudaMalloc((void**)&col_d, numCols * sizeof(int)));
    CHECK(cudaMalloc((void**)&val_d, nnz * sizeof(float)));

    CHECK(cudaMalloc((void**)&dnsMtxB_d, numRows_B * numCols_B * sizeof(float)));
    CHECK(cudaMalloc((void**)&dnsMtxAB_d, numRows * numCols_B * sizeof(float)));


    //(2) Copy value to device
    CHECK(cudaMemcpy(row_d, row, (numRows+1) *sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(col_d, col, numCols * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(val_d, val, nnz * sizeof(int), cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(dnsMtxB_d, denseMtxB, numRows_B * numCols_B * sizeof(float), cudaMemcpyHostToDevice));
    

    //(3) Create cuspare handle and descreptors
    cusparseSpMatDescr_t mtxA_dscr;
    cusparseDnMatDescr_t mtxB_dscr, mtxC_dscr;

    checkCudaErrors(cusparseCreateCsr(&mtxA_dscr, numRows, numCols, nnz, row_d, col_d, val_d,CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    checkCudaErrors(cusparseCreateDnMat(&mtxB_dscr, numRows_B, numCols_B, numRows_B, dnsMtxB_d, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    checkCudaErrors(cusparseCreateDnMat(&mtxC_dscr, numRows, numCols_B, numRows, dnsMtxAB_d, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    //(4) Computer sparse-dense matrix multiplication



    return 0;
} // end of main

