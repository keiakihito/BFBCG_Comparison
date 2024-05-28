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

//Bigger size matrix


// Define the dense matrixB
float mtxA[] = {
    1.0, 1.0,
    0.0, 1.0, 
    1.0, 0.0
};

#define ROW_A 3 
#define COL_A 2

//Leading dimension of the matrix A
//Set to the number of rows of the matrix 
//when it is stored in column-major order.
#define LD_A 3

// Time tracker for each iteration
double myCPUTimer()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}

int main(int argc, char** argv){

    //FIXME: It excutes, but the value are wrong.😫😫😫

    //(1) Allocate device memory
    float *mtxA_d = NULL;
    float *mtxU_d = NULL;
    float *mtxD_d = NULL; // Singular values
    float *mtxVT_d = NULL;

    /*The devInfo is an integer pointer
    It points to device memory where cuSOLVER can store information 
    about the success or failure of the computation.*/
    int *devInfo = NULL;

    int lwork = 0;//Size of workspace
    //work_d is a pointer to device memory that serves as the workspace for the computation
    //Then passed to the cuSOLVER function performing the computation.
    float *work_d = NULL; // 
    float *rwork_d = NULL; // Place holder


    //Specifies options for computing all or part of the matrix U: = ‘A’: 
    //all m columns of U are returned in array
    signed char jobU = 'A';

    //Specifies options for computing all or part of the matrix V**T: = ‘A’: 
    //all N rows of V**T are returned in the array
    signed char jobVT = 'A';

    bool debug = false;

    CHECK((cudaMalloc((void**)&mtxA_d, ROW_A * COL_A *sizeof(float))));
    // Set up size of matrix U which is 3 by 3 for Full SVD
    CHECK((cudaMalloc((void**)&mtxU_d, LD_A * ROW_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxD_d, COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&mtxVT_d, LD_A * COL_A * sizeof(float))));
    //Should be the same size of VT in this case 2 by 2
    // CHECK((cudaMalloc((void**)&mtxTemp_d, LD_A * COL_A * sizeof(float))));
    CHECK((cudaMalloc((void**)&devInfo, sizeof(int))));
    

    //(2) Copy value to device
    CHECK((cudaMemcpy(mtxA_d, mtxA, ROW_A * COL_A * sizeof(float), cudaMemcpyHostToDevice)));

    debug = false;
    if(debug){
        printf("\n\n~~~MtxA~~\n");
        print_mtx_d(mtxA_d, ROW_A, COL_A);
    }
    debug = false;

    //(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));

    checkCudaErrors(cusolverDnSgesvd_bufferSize(cusolverHandler, ROW_A, COL_A, &lwork));
    CHECK(cudaMalloc((void**)&work_d, lwork * sizeof(float)));

    //(4) Compute SVD decomposition
    checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, ROW_A, COL_A, mtxA_d, LD_A, mtxD_d, mtxU_d,LD_A, mtxVT_d, LD_A, work_d, lwork, rwork_d, devInfo));

    debug = true;
    if(debug){
        printf("\n\n~~mtxU_d\n");
        print_mtx_d(mtxU_d, ROW_A, COL_A);
        printf("\n\n~~mtxD_d\n");
        print_mtx_d(mtxD_d, COL_A, 1);
        printf("\n\n~~mtxVT_d\n");
        print_mtx_d(mtxVT_d, COL_A, COL_A);
    }
    debug = false;

    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxA_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(mtxD_d));
    CHECK(cudaFree(mtxVT_d));
    CHECK(cudaFree(devInfo));
    CHECK(cudaFree(work_d));
    CHECK(cudaFree(rwork_d));

    return 0;
}// end of main
