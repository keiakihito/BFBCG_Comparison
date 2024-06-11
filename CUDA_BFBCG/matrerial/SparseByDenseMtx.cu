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
    float val[] = {10.0, 1.0, 1.0, 20.0, 1.0, 1.0, 30.0, 1.0, 1.0, 40.0, 1.0, 1.0, 50.0};

    //For sparse matrix A
    int numRows = 5;
    int numCols = 5;
    int nnz = 13;// Number of Non zero

    //For dense matrix B
    int numRows_B = 5;
    int numCols_B = 3;

    float alpha = 1.0;
    float beta = 0.0;

    //(1) Allocate device memory
    int *row_d = NULL;
    int *col_d = NULL;
    float *val_d = NULL;

    float *dnsMtxB_d = NULL;
    float *dnsMtxAB_h = NULL;// Result in host
    float *dnsMtxAB_d = NULL;// Result in device

    bool debug = false;

    CHECK(cudaMalloc((void**)&row_d, (numRows+1) * sizeof(int)));
    CHECK(cudaMalloc((void**)&col_d, nnz * sizeof(int)));
    CHECK(cudaMalloc((void**)&val_d, nnz * sizeof(float)));
    

    CHECK(cudaMalloc((void**)&dnsMtxB_d, numRows_B * numCols_B * sizeof(float)));
    CHECK(cudaMalloc((void**)&dnsMtxAB_d, numRows * numCols_B * sizeof(float)));


    //(2) Copy value to device
    CHECK(cudaMemcpy(row_d, row, (numRows+1) *sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(col_d, col, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(val_d, val, nnz * sizeof(float), cudaMemcpyHostToDevice));

    debug = false;
    if(debug){
        printf("\n\nrow vector \n");
        print_vector(row_d, numRows+1);
        printf("\n\ncol vector \n");
        print_vector(col_d, nnz);
        printf("\n\nval vector \n");
        print_vector(val_d, nnz);
    }
    debug = false;


    CHECK(cudaMemcpy(dnsMtxB_d, denseMtxB, numRows_B * numCols_B * sizeof(float), cudaMemcpyHostToDevice));
    // print_mtx_d(dnsMtxB_d, numRows_B, numCols_B);   
    // print_mtx_d(dnsMtxAB_d, numRows, numCols_B);    

    //(3) Create cuspare handle and descreptors
    cusparseSpMatDescr_t mtxA_dscr;
    cusparseDnMatDescr_t mtxB_dscr, mtxC_dscr;

    cusparseHandle_t cusparseHandle;
    cusparseCreate(&cusparseHandle);

    
    checkCudaErrors(cusparseCreateCsr(&mtxA_dscr, numRows, numCols, nnz, row_d, col_d, val_d,CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    //Note Given the marix is row major order (CUSPARSE_ORDER_ROW), the leading dimension is number of column.
    checkCudaErrors(cusparseCreateDnMat(&mtxB_dscr, numRows_B, numCols_B, numCols_B, dnsMtxB_d, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    checkCudaErrors(cusparseCreateDnMat(&mtxC_dscr, numRows, numCols_B, numCols_B, dnsMtxAB_d, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    
    debug = false;
    if(debug){
        printf("\n\nrow vector \n");
        print_vector(row_d, numRows+1);
        printf("\n\ncol vector \n");
        print_vector(col_d, nnz);
        printf("\n\nval vector \n");
        print_vector(val_d, nnz);

        printf("\n\ndnsMtxB\n");
        print_mtx_d(dnsMtxB_d, numRows_B, numCols_B);
        printf("\n\ndnsMtxAB\n");
        print_mtx_d(dnsMtxAB_d, numRows, numCols_B);   
    }
    debug = false;

 

    //(4) Computer sparse-dense matrix multiplication

    //Need to allocate buffer for cusparseSpMM
    size_t bufferSize = 0;
    void* dBuffer = NULL;
    checkCudaErrors(cusparseSpMM_bufferSize(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, mtxA_dscr, mtxB_dscr, &beta, mtxC_dscr, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));

    CHECK(cudaMalloc(&dBuffer, bufferSize));

    //Perform sparse * dense matrix operaroin
    checkCudaErrors(cusparseSpMM(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha, mtxA_dscr, mtxB_dscr, &beta, mtxC_dscr, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));


  

    // (5) Copy back the result to host
    dnsMtxAB_h = (float*)malloc(sizeof(float)* numRows * numCols_B);
    CHECK(cudaMemcpy(dnsMtxAB_h, dnsMtxAB_d, sizeof(float)*(numRows * numCols_B), cudaMemcpyDeviceToHost));

    printf("\n\n~~Check sprMtxA * dnsMtxB~~\n");
    print_mtx_h(dnsMtxAB_h, numRows, numCols_B);

    //(6) Free pointers
    checkCudaErrors(cusparseDestroySpMat(mtxA_dscr));
    checkCudaErrors(cusparseDestroyDnMat(mtxB_dscr));
    checkCudaErrors(cusparseDestroyDnMat(mtxC_dscr));
    checkCudaErrors(cusparseDestroy(cusparseHandle));

    CHECK(cudaFree(dBuffer));
    CHECK(cudaFree(row_d));
    CHECK(cudaFree(col_d));
    CHECK(cudaFree(val_d));
    CHECK(cudaFree(dnsMtxB_d));
    CHECK(cudaFree(dnsMtxAB_d));
    
    free(dnsMtxAB_h);

    return 0;
} // end of main

