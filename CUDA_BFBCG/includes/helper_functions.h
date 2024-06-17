#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <iostream>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cmath>
#include<sys/time.h>

// helper function CUDA error checking and initialization
#include "helper_cuda.h"  
#include "helper_debug.h"
#include "CSRMatrix.h"  

#define CHECK(call){ \
    const cudaError_t cuda_ret = call; \
    if(cuda_ret != cudaSuccess){ \
        printf("Error: %s:%d,  ", __FILE__, __LINE__ );\
        printf("code: %d, reason: %s \n", cuda_ret, cudaGetErrorString(cuda_ret));\
        exit(-1); \
    }\
}

//function headers
//Generate random SPD dense matrix
// N is matrix size
// SPD <- mtxA * mtxA'
float* generateSPD_DenseMatrix(int N);

// N is matrix size
float* generate_TriDiagMatrix(int N);

//Dense Diagonally doinat SPD matrix 
float* generateWellConditoinedSPDMatrix(int N);


void validateSol(const float *mtxA_h, const float* x_h, float* rhs, int N);

//Input: cublasHandle_t cublasHandler, float* matrix Residual, int number of row, int number of column
//Process: Extracts the first column vector from the residual matrix,
// 			Calculate dot product of the first column vector, then compare sqare root of dot product with Threashold
//Output: boolean
bool checkStop(cublasHandle_t cublasHandler, float *mtxR_d, int numOfRow, int numOfClm, float const threshold);

//Inverse funtion with sub functions
//Input: float* matrix A, float* matrix A inverse, int N, number of Row or Column
//Process: Inverse matrix
//Output: lfoat mtxQPT_d
void inverse_Den_Mtx(cusolverDnHandle_t cusolverHandler, float* mtxA_d, float* mtxA_inv_d, int N);


//Input: float* identity matrix, int numer of Row, Column
//Process: Creating identity matrix with number of N
//Output: float* mtxI
__global__ void identity_matrix(float* mtxI_d, int N);

//Input: float* identity matrix, int N, which is numer of Row or Column
//Process: Call the kernel to create identity matrix with number of N
//Output: float* mtxI
void createIdentityMtx(float* mtxI_d, int N);


//Input: const CSRMatrix &csrMtx, float *dnsMtxB_d, int numClmB, float * dnxMtxC_d
//Process: Matrix Multiplication Sparse matrix and Dense matrix
//Output: dnsMtxC_d, dense matrix C in device
void multiply_Sprc_Den_mtx(const CSRMatrix &csrMtx, float *dnsMtxB_d, int numClmsB, float * dnsMtxC_d);






//Input: float* mtxZ, int number of Row, int number Of column, int & currentRank
//Process: the function extracts orthonormal set from the matrix Z
//Output: float* mtxY_hat, the orthonormal set of matrix Z.
float* orth(float* mtxZ_d, int numOfRow, int numOfClm, int &currentRank);

//Input: cusolverDnHandler, int number of row, int number of column, int leading dimensino, 
//		 float* matrix A, float* matrix U, float* vector singlluar values, float* matrix V tranpose
//Process: Singluar Value Decomposion
//Output: float* Matrix U, float* singular value vectors, float* Matrix V transpose
void SVD_Decmp(cusolverDnHandle_t cusolverHandler, int numOfRow, int numOfClm, int ldngDim, float* mtxA_d, float* mtxU_d, float* sngVals_d, float*mtxVT_d);

//Input: singluar values, int currnet rank, float threashould
//Process: Check eigenvalues are greater than threashould, then set new rank
//Output: int newRank
int setRank(float* sngVals_d, int currentRank, float threashold);

//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, float* matrix A in device, float* matrix B in device , float* result matrix C in device, int leading dimension A, in leading dimension B
//Process: matrix multiplication matrix A and matrix B
//Result: matrix C as a result
void multiply_Den_ClmM_mtx_mtx(cublasHandle_t cublasHandler, float* mtxA_d, float* mtxB_d, float* mtxC_d, int numOfRowA, int numOfColB, int numOfColA);

//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, float* matrix A in device, float* matrix B in device , float* result matrix C in device, int number of Row, int number of column
//Process: matrix multiplication matrix A' * matrix A
//Result: matrix C as a result with square matrix
void multiply_Den_ClmM_mtxT_mtx(cublasHandle_t cublasHandler, float* mtxA_d, float* mtxC_d, int numOfRow, int numOfClm);

//Input matrix should be column major, overload function
//Input: cubasHandle_t cublasHandler, float* matrix A in device, float* matrix B in device , float* result matrix C in device, int number of Row A, int number of column A, int number of cloumn B
//Process: matrix multiplication matrix A' * matrix A
//Result: matrix C as a result with square matrix
void multiply_Den_ClmM_mtxT_mtx(cublasHandle_t cublasHandler, float* mtxA_d, float* mtxB_d, float* mtxC_d, int numOfRowA, int numOfClmA, int numOfClmB);


//Input: cublasHandler_t cublasHandler, float* matrix X, int number of row, int number of column
//Process: the function allocate new memory space and tranpose the mtarix X
//Output: float* matrix X transpose
float* transpose_Den_Mtx(cublasHandle_t cublasHandler, float* mtxX_d, int numOfRow, int numOfClm);

//Input: float* matrix V, int number of Row and Column, int current rank
//Process: the functions truncates the matrix V with current rank
//Output: float* matrix V truncated.
float* truncate_Den_Mtx(float* mtxV_d, int numOfN, int currentRank);

//Input: float* mtxY, product of matrix Z * matrix U, int number of row, int number of column 
//Process: the kernel normalize each column vector of matrix Y in 2 norm
//Output: float* mtxY_d, which will be updated as normalized matrix Y hat.
__global__ void normalizeClmVec(float* mtxY_d, int numOfRow, int numOfCol);


//Input: float* mtxY, product of matrix Z * matrix U, int number of row, int number of column 
//Process: the function calls kernel and normalize each column vector 
//Output: float* mtxY_d, which will be updated as normalized matrix Y hat.
void normalize_Den_Mtx(float* mtxY_d, int numOfRow, int numOfCol);










//Input: float* matrix A, int number of Row, int number Of Column
//Process: Compute condition number and check whther it is ill-conditioned or not.
//Output: float condition number
float computeConditionNumber(float* mtxA_d, int numOfRow, int numOfClm);

//Input: cusolverDnHandle_t cusolverHandler, int number of row, int number of column, int leading dimension, float* matrix A
//Process: Extract eigenvalues with full SVD
//Output: float* sngVals_d, singular values in device in column vector
float* extractSngVals(cusolverDnHandle_t cusolverHandler, int numOfRow, int numOfClm, int ldngDim, float* mtxA_d);







// = = = Function signatures = = = = 
//Generate random SPD dense matrix
// N is matrix size
float* generateSPD_DenseMatrix(int N){
	float* mtx_h = NULL;
	float* mtx_d = NULL;
	float* mtxSPD_h = NULL;
	float* mtxSPD_d = NULL;

	//Using for cublas function
	const float alpha = 1.0f;
	const float beta = 0.0f;

	//Allocate memoery in Host
	mtx_h = (float*)malloc(sizeof(float)* (N*N));
	mtxSPD_h = (float*)malloc(sizeof(float)* (N*N));

	if(! mtx_h || ! mtxSPD_h){
		printf("\nFailed to allocate memory in host\n\n");
		return NULL;
	}

	// Seed the random number generator
	srand(static_cast<unsigned>(time(0)));

	// Generate and store to mtx_h in all elements random values between 0 and 1.
	for (int wkr = 0; wkr < N*N;  wkr++){
		if(wkr % N == 0){
			printf("\n");
		}
		mtx_h[wkr] = ((float)rand()/RAND_MAX);
		// printf("\nmtx_h[%d] = %f", wkr, mtx_h[wkr]);

	}



	//(1)Allocate memoery in device
	CHECK(cudaMalloc((void**)&mtx_d, sizeof(float) * (N*N)));
	CHECK(cudaMalloc((void**)&mtxSPD_d, sizeof(float) * (N*N)));

	//(2) Copy value from host to device
	CHECK(cudaMemcpy(mtx_d, mtx_h, sizeof(float)* (N*N), cudaMemcpyHostToDevice));

	//(3) Calculate SPD matrix <- A' * A
	// Create a cuBLAS handle
	cublasHandle_t handle;
	cublasCreate(&handle);
	checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, N, &alpha, mtx_d, N, mtx_d, N, &beta, mtxSPD_d, N));

	//(4) Copy value from device to host
	CHECK(cudaMemcpy(mtxSPD_h, mtxSPD_d, sizeof(float) * (N*N), cudaMemcpyDeviceToHost));
	
	//(5) Free memeory
	cudaFree(mtx_d);
	cudaFree(mtxSPD_d);
	cublasDestroy(handle);
	free(mtx_h);


	return mtxSPD_h;
} // enf of generateSPD_DenseMatrix


// N is matrix size
float* generate_TriDiagMatrix(int N)
{
	//Allocate memoery in Host
	float* mtx_h = (float*)calloc(N*N, sizeof(float));


	if(! mtx_h){
		printf("\nFailed to allocate memory in host\n\n");
		return NULL;
	}

	// Seed the random number generator
	srand(static_cast<unsigned>(time(0)));

	// Generate and store to mtx_h tridiagonal random values between 0 and 1.
	mtx_h[0] = ((float)rand()/RAND_MAX)+10.0f;
	mtx_h[1] = ((float)rand()/RAND_MAX);
	for (int wkr = 1; wkr < N -1 ;  wkr++){
		mtx_h[(wkr * N) + (wkr-1)] =((float)rand()/RAND_MAX);
		mtx_h[wkr * N + wkr] = ((float)rand()/RAND_MAX)+10.0f;
		mtx_h[(wkr * N) + (wkr+1)] = ((float)rand()/RAND_MAX);
	}
	mtx_h[(N*(N-1))+ (N-2)] = ((float)rand()/RAND_MAX);
	mtx_h[(N*(N-1)) + (N-1)] = ((float)rand()/RAND_MAX) + 10.0f;
	
	// printf("\nTridiagonal Mtx: \n");
	// for(int rw_wkr = 0; rw_wkr < N; rw_wkr++){
	// for(int clm_wkr = 0; clm_wkr < N; clm_wkr++){
	// 	printf("%f ", mtx_h[rw_wkr*N + clm_wkr]);
	// }
	// printf("\n");
	// }
	// printf("\n\n");

	//Scale down
	for (int i = 0; i < N * N; i++) {
        mtx_h[i] /= 10;
    }

	return mtx_h;
} // enf of generate_TriDiagMatrix


float* generateWellConditoinedSPDMatrix(int N)
{
	float* mtxA_h = (float*)malloc(N * N * sizeof(float));

	srand(time(NULL));

	//Generate a random matrix
	for (int otWkr = 0; otWkr < N; otWkr++){
		for (int inWkr = 0; inWkr <= otWkr; inWkr++){
			float val = (float)rand() / RAND_MAX;
			mtxA_h[otWkr * N + inWkr] = val;
			mtxA_h[inWkr * N + otWkr] = val;
		} // end of inner loop
	} // end of outer loop

	//Ensure the matrix is diagonally dominant
	for (int wkr = 0; wkr < N; wkr++){
		mtxA_h[wkr * N + wkr] += N;
	}

	return mtxA_h;
} // end of generateWellConditoinedSPDMatrix

void validateSol(const float *mtxA_h, const float* x_h, float* rhs, int N)
{
    float rsum, diff, error = 0.0f;

    for (int rw_wkr = 0; rw_wkr < N; rw_wkr++){
        rsum = 0.0f;
        for(int clm_wkr = 0; clm_wkr < N; clm_wkr++){
            rsum += mtxA_h[rw_wkr*N + clm_wkr]* x_h[clm_wkr];
            // printf("\nrsum = %f", rsum);
        } // end of inner loop
        diff = fabs(rsum - rhs[rw_wkr]);
        if(diff > error){
            error = diff;
        }
        
    }// end of outer loop
    
    printf("\n\nTest Summary: Error amount = %f\n", error);

}// end of validateSol





//Stop condition
//TO DO implement and test isStop, inverse, and sparseMul
//Input: cublasHandle_t cublasHandler, float* matrix Residual, int number of row, int number of column
//Process: Extracts the first column vector from the residual matrix,
// 			Calculate dot product of the first column vector, then compare sqare root of dot product with Threashold
bool checkStop(cublasHandle_t cublasHandler, float *mtxR_d, int numOfRow, int numOfClm, float const threshold)
{
	float* r1_d = NULL;
	float dotPrdct = 0.0f;
	bool debug =false;

	//Extract first column
	CHECK(cudaMalloc((void**)&r1_d, numOfRow * sizeof(float)));
	CHECK(cudaMemcpy(r1_d, mtxR_d, numOfRow * sizeof(float), cudaMemcpyDeviceToDevice));

	if(debug){
		printf("\n\nvector r_1: \n");
		print_vector(r1_d, numOfRow);
	}
	
	//Dot product of r_{1}' * r_{1}, cublasSdot
	checkCudaErrors(cublasSdot(cublasHandler, numOfRow, r1_d, 1, r1_d, 1, &dotPrdct));

	//Square root(dotPrdct)
	if(debug){
		printf("\n\ndot product of r_1: %.10f\n", dotPrdct);
		printf("\n\nsqrt(dot product of r_1): %.10f\n", sqrt(dotPrdct));
		printf("\n\nTHRESHOLD : %f\n", threshold);
	}

	return (sqrt(dotPrdct)< threshold);
}

//Inverse
//Input: float* matrix A, float* matrix A inverse, int N, number of Row or Column
//Process: Inverse matrix
//Output: lfoat mtxQPT_d
void inverse_Den_Mtx(cusolverDnHandle_t cusolverHandler, float* mtxA_d, float* mtxA_inv_d, int N)
{
	float* mtxA_cpy_d = NULL;

	float *work_d = nullptr;

    //The devInfo pointer holds the status information after the LU decomposition or solve operations.
    int *devInfo = nullptr;
    
    /*
    A pivots_d pointer holds the pivot indices generated by the LU decomposition. 
    These indices indicate how the rows of the matrix were permuted during the factorization.
    */
    int *pivots_d = nullptr;
    
    //Status information specific to the LAPACK operations performed by cuSolver.
    // int *lapackInfo = nullptr;

    // Size of the workng space
    int lwork = 0;
	bool debug = true;


    if(debug){
        printf("\n\n~~mtxA_d~~\n\n");
        print_mtx_clm_d(mtxA_d, N, N);
    }

	//(1) Make copy of mtxA
	CHECK(cudaMalloc((void**)&mtxA_cpy_d, N * N * sizeof(float)));
	CHECK(cudaMemcpy(mtxA_cpy_d, mtxA_d, N * N * sizeof(float), cudaMemcpyDeviceToDevice));
	
	if(debug){
		printf("\n\n~~mtxA_cpy_d~~\n\n");
		print_mtx_clm_d(mtxA_cpy_d, N, N);
	}

	//(2) Create Identity matrix
	createIdentityMtx(mtxA_inv_d, N);
	if(debug){
		printf("\n\n~~mtxI~~\n\n");
		print_mtx_clm_d(mtxA_inv_d, N, N);
	}

	//(3)Calculate work space for cusolver
    checkCudaErrors(cusolverDnSgetrf_bufferSize(cusolverHandler, N, N, mtxA_cpy_d, N, &lwork));
    checkCudaErrors(cudaMalloc((void**)&work_d, lwork * sizeof(float)));
	CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));
    CHECK(cudaMalloc((void**)&pivots_d, N * sizeof(int)));

	//(4.1) Perform the LU decomposition, 
    checkCudaErrors(cusolverDnSgetrf(cusolverHandler, N, N, mtxA_cpy_d, N, work_d, pivots_d, devInfo));
    cudaDeviceSynchronize();

	//Check LU decomposition was successful or not.
	//If not, it can be ill-conditioned or singular.
	int devInfo_h;
	checkCudaErrors(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if(devInfo_h != 0){
		printf("\n\nLU decomposition failed in the inverse_Den_Mtx, info = %d\n", devInfo_h);
		if(devInfo_h == 11){
			printf("\n!!!The matrix potentially is ill-conditioned or singular!!!\n\n");
			float* mtxA_check_d = NULL;
			CHECK(cudaMalloc((void**)&mtxA_check_d, N * N * sizeof(float)));
			CHECK(cudaMemcpy(mtxA_check_d, mtxA_d, N * N * sizeof(float), cudaMemcpyDeviceToDevice));
			float conditionNum = computeConditionNumber(mtxA_check_d, N, N);
			printf("\n\n🔍Condition number = %f🔍\n\n", conditionNum);
			CHECK(cudaFree(mtxA_check_d));
		}
		exit(1);
	}


    /*
    mtxA_d will be a compact form such that 

    A = LU = | 4  1 |
             | 1  3 |
    
    L = |1    0 |  U = |4    1  |
        |0.25 1 |      |0   2.75|
    
    mtxA_d compact form = | 4      1  |
                          | 0.25  2.75|
    */
	if(debug){
        printf("\n\nAfter LU factorization\n");
        printf("\n\n~~mtxA_cpy_d~~\n");
        print_mtx_clm_d(mtxA_cpy_d, N, N);
    }


    //(4.2)Solve for the iverse, UX = Y
    /*
    A = LU
    A * X = LU * X = I
    L * (UX) = L * Y = I
    UX = Y
    */
    checkCudaErrors(cusolverDnSgetrs(cusolverHandler, CUBLAS_OP_N, N, N, mtxA_cpy_d, N, pivots_d, mtxA_inv_d, N, devInfo));
	cudaDeviceSynchronize();

	//Check solver after LU decomposition was successful or not.
	checkCudaErrors(cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if(devInfo_h != 0){
		printf("Solve after LU failed, info = %d\n", devInfo_h);
		exit(1);
	}


	//(5)Free memoery
	CHECK(cudaFree(mtxA_cpy_d));
	CHECK(cudaFree(work_d));
    CHECK(cudaFree(devInfo));
    CHECK(cudaFree(pivots_d));

} // end of inverse_Den_Mtx


//Input: float* identity matrix, int numer of Row, Column
//Process: Creating dense identity matrix with number of N
//Output: float* mtxI
__global__ void identity_matrix(float* mtxI_d, int N)
{	
	//Get global index 
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	//Set boundry condition
	if(idx < (N * N)){
		int glbRow = idx / N;
		int glbClm = idx % N;

		// Index points to the diagonal element
		if(glbRow == glbClm){
			mtxI_d[idx] = 1.0f;
		}else{
			mtxI_d[idx] = 0.0f;
		}// end of if, store diagonal element

	} // end of if, boundtry condition

}// end of identity_matrix


//Input: float* identity matrix, int N, which is numer of Row or Column
//Process: Call the kernel to create identity matrix with number of N
//Output: float* mtxI
void createIdentityMtx(float* mtxI_d, int N)
{		

	// Use a 1D block and grid configuration
    int blockSize = 1024; // Number of threads per block
    int gridSize = ceil((float)N * N / blockSize); // Number of blocks needed

    identity_matrix<<<gridSize, blockSize>>>(mtxI_d, N);
    
	cudaDeviceSynchronize(); // Ensure the kernel execution completes before proceeding
}









//Sparse matrix multiplicatation
//Input: const CSRMatrix &csrMtx, float *dnsMtxB_d, int numClmB, float * dnxMtxC_d
//Process: Matrix Multiplication Sparse matrix and Dense matrix
//Output: dnsMtxC_d, dense matrix C in device
void multiply_Sprc_Den_mtx(const CSRMatrix &csrMtx, float *dnsMtxB_d, int numClmsB, float * dnsMtxC_d)
{
	int numRowsA = csrMtx.numOfRows;
	int numClmsA = csrMtx.numOfClms;
	int nnz = csrMtx.numOfnz;

	float alpha = 1.0f;
	float beta = 0.0f;

	bool debug = false;


	//(1) Allocate device memoery for CSR matrix
	int	*row_offsets_d = NULL;
	int *col_indices_d = NULL;
	float *vals_d = NULL;

	CHECK(cudaMalloc((void**)&row_offsets_d, (numRowsA + 1) * sizeof(int)));
	CHECK(cudaMalloc((void**)&col_indices_d, nnz * sizeof(int)));
	CHECK(cudaMalloc((void**)&vals_d, nnz * sizeof(float)));

	//(2) Copy values from host to device
	CHECK(cudaMemcpy(row_offsets_d, csrMtx.row_offsets, (numRowsA+1) * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(col_indices_d, csrMtx.col_indices, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(vals_d, csrMtx.vals, nnz * sizeof(float), cudaMemcpyHostToDevice));

	//(3) Crate cuSPARSE handle and descriptors
	cusparseHandle_t cusparseHandler;
	cusparseCreate(&cusparseHandler);

	cusparseSpMatDescr_t mtxA;
	cusparseDnMatDescr_t mtxB, mtxC;

	checkCudaErrors(cusparseCreateCsr(&mtxA, numRowsA, numClmsA, nnz, row_offsets_d, col_indices_d, vals_d, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
	checkCudaErrors(cusparseCreateDnMat(&mtxB, numClmsA, numClmsB, numClmsA, dnsMtxB_d, CUDA_R_32F, CUSPARSE_ORDER_COL));
	checkCudaErrors(cusparseCreateDnMat(&mtxC, numRowsA, numClmsB, numRowsA, dnsMtxC_d, CUDA_R_32F, CUSPARSE_ORDER_COL));

	//(4) Calculate buffer size of Spase by dense matrix mulply operation
    size_t bufferSize = 0;
    void *dBuffer = NULL;
	checkCudaErrors(cusparseSpMM_bufferSize(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mtxA, mtxB, &beta, mtxC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
	CHECK(cudaMalloc(&dBuffer, bufferSize));

	//(5)Perform sparse-dense matrix Multiplication
	checkCudaErrors(cusparseSpMM(cusparseHandler, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mtxA, mtxB, &beta, mtxC, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

	if(debug){
		printf("\n\n~~mtxC after cusparseSpMM~~\n\n");
		print_mtx_clm_d(dnsMtxC_d, numRowsA, numClmsB);
	}

	//(6) Free memeory and destroy descriptors
	checkCudaErrors(cusparseDestroySpMat(mtxA));
	checkCudaErrors(cusparseDestroyDnMat(mtxB));
	checkCudaErrors(cusparseDestroyDnMat(mtxC));
	checkCudaErrors(cusparseDestroy(cusparseHandler));

	CHECK(cudaFree(dBuffer));
	CHECK(cudaFree(row_offsets_d));
	CHECK(cudaFree(col_indices_d));
	CHECK(cudaFree(vals_d));

} // end of multiply_Src_Den_mtx








//Orth functions
//Input: float* mtxZ, int number of Row, int number Of column, int & currentRank
//Process: the function extracts orthonormal set from the matrix Z
//Output: float* mtxY_hat, the orthonormal set of matrix Z.
float* orth(float* mtxZ_d, int numOfRow, int numOfClm, int &currentRank)
{	
	/*
	Pseudocode
	// Mayby need to make a copy of mtxZ
	Transpose Z
	Multiply mtxS <- mtxZ' * mtxZ
	Perform SVD
	Transpoze mtxVT, and get mtxV
	Call set Rank
	if(newRank < currentRank){
		Trancate mtxV
		currentRank <- newRank
	}else{
		continue;
	}
	Mutiply mtxY <- mtxZ * mtxV 
	Normalize mtxY <- mtxY
	Return mtxY
	*/

	float *mtxY_d = NULL; // Orthonormal set, serach diretion
	float *mtxZ_cpy_d = NULL; // Need a copy to tranpose mtxZ'
	float *mtxS_d = NULL;

	float *mtxU_d = NULL;
	float *sngVals_d = NULL;
	float *mtxV_d = NULL;
	float *mtxVT_d = NULL;
	float *mtxV_trnc_d = NULL;

	const float THREASHOLD = 1e-5;

	bool debug = true;




	if(debug){
		printf("\n\n~~mtxZ~~\n\n");
		print_mtx_clm_d(mtxZ_d, numOfRow, numOfClm);
	}


	//(1) Allocate memeory in device
	//Make a copy of mtxZ for mtxZ'
    CHECK(cudaMalloc((void**)&mtxZ_cpy_d, numOfRow * numOfClm * sizeof(float)));
	CHECK(cudaMalloc((void**)&mtxS_d, numOfRow * numOfClm * sizeof(float)));
	
	//For SVD decomposition
	CHECK(cudaMalloc((void**)&mtxU_d, numOfRow * numOfClm * sizeof(float)));
	CHECK(cudaMalloc((void**)&sngVals_d, numOfClm * sizeof(float)));
	CHECK(cudaMalloc((void**)&mtxVT_d, numOfClm * numOfClm * sizeof(float)));

	//(2) Copy value to device
	CHECK(cudaMemcpy(mtxZ_cpy_d, mtxZ_d, numOfRow * numOfClm * sizeof(float), cudaMemcpyDeviceToDevice));
	
	
	if(debug){
		printf("\n\n~~mtxZ cpy~~\n\n");
		print_mtx_clm_d(mtxZ_cpy_d, numOfRow, numOfClm);
	}

	//(3) Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    cublasHandle_t cublasHandler = NULL;

    checkCudaErrors(cusolverDnCreate(&cusolverHandler));
    checkCudaErrors(cublasCreate(&cublasHandler));

	//(4) Perform orthonormal set prodecure
	//(4.1) Mutiply mtxS <- mtxZ' * mtxZ
	//mtxZ_cpy will be free after multiplication
	multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxZ_cpy_d, mtxS_d, numOfRow, numOfClm);

	if(debug){
		printf("\n\n~~mtxS ~~\n\n");
		print_mtx_clm_d(mtxS_d, numOfClm, numOfClm);
	}

	//(4.2)SVD Decomposition
	SVD_Decmp(cusolverHandler, numOfClm, numOfClm, numOfClm, mtxS_d, mtxU_d, sngVals_d, mtxVT_d);
	if(debug){
		printf("\n\n~~mtxU ~~\n\n");
		print_mtx_clm_d(mtxU_d, numOfClm, numOfClm);
		printf("\n\n~~sngVals ~~\n\n");
		print_mtx_clm_d(sngVals_d, numOfClm, 1);
		printf("\n\n~~mtxVT ~~\n\n");
		print_mtx_clm_d(mtxVT_d, numOfClm, numOfClm);
	}	

	//(4.3) Transpose mtxV <- mtxVT'
	//mtxVT_d will be free inside function
	mtxV_d = transpose_Den_Mtx(cublasHandler, mtxVT_d, numOfClm, numOfClm);
	if(debug){
		printf("\n\n~~mtxV ~~\n\n");
		print_mtx_clm_d(mtxV_d, numOfClm, numOfClm);
	}	

	//(4.4) Set current rank
	currentRank = setRank(sngVals_d, currentRank, THREASHOLD);
	if(debug){
		printf("\n\n~~ new rank = %d ~~\n\n", currentRank);
	}

	//(4.5) Truncate matrix V
	//mtxV_d will be free after truncate_Den_Mtx
	mtxV_trnc_d = truncate_Den_Mtx(mtxV_d, numOfClm, currentRank);
		
	if(debug){
		printf("\n\n~~mtxV_Trnc ~~\n\n");
		print_mtx_clm_d(mtxV_trnc_d, numOfClm, currentRank);
	}	

	//(4.6) Multiply matrix Y <- matrix Z * matrix V Truncated
	CHECK(cudaMalloc((void**)&mtxY_d, numOfRow * currentRank * sizeof(float)));
	multiply_Den_ClmM_mtx_mtx(cublasHandler, mtxZ_d, mtxV_trnc_d, mtxY_d, numOfRow, currentRank, numOfClm);
	
	if(debug){
		printf("\n\n~~mtxY ~~\n\n");
		print_mtx_clm_d(mtxY_d, numOfRow, currentRank);
	}

	//(4.7) Normalize matrix Y_hat <- normalize_Den_Mtx(mtxY_d)
	normalize_Den_Mtx(mtxY_d, numOfRow, currentRank);
	if(debug){
		printf("\n\n~~mtxY hat <- orth(*) ~~\n\n");
		print_mtx_clm_d(mtxY_d, numOfRow, currentRank);
	}

	//(4.6) Check orthogonality
	if(debug){
		//Check the matrix Y hat column vectors are orthogonal eachother
		float* mtxI_d = NULL;
		CHECK(cudaMalloc((void**)&mtxI_d, currentRank * currentRank * sizeof(float)));
		multiply_Den_ClmM_mtxT_mtx(cublasHandler, mtxY_d, mtxI_d, numOfRow, currentRank);
		printf("\n\n~~~~Orthogonality Check (should be close to identity matrix)~~\n\n");
		print_mtx_clm_d(mtxI_d, currentRank, currentRank);
		CHECK(cudaFree(mtxI_d));
	}

	//(5) Free memory
    checkCudaErrors(cusolverDnDestroy(cusolverHandler));
    checkCudaErrors(cublasDestroy(cublasHandler));

    CHECK(cudaFree(mtxZ_d));
	CHECK(cudaFree(mtxS_d));
    CHECK(cudaFree(mtxU_d));
    CHECK(cudaFree(sngVals_d));
    CHECK(cudaFree(mtxV_trnc_d));

	return mtxY_d;
}

//Input: cusolverDnHandler, int number of row, int number of column, int leading dimensino, 
//		 float* matrix A, float* matrix U, float* vector singlluar values, float* matrix V tranpose
//Process: Singluar Value Decomposion
//Output: float* Matrix U, float* singular value vectors, float* Matrix V transpose
void SVD_Decmp(cusolverDnHandle_t cusolverHandler, int numOfRow, int numOfClm, int ldngDim, float* mtxA_d, float* mtxU_d, float* sngVals_d, float*mtxVT_d)
{	
	//Make a copy of matrix A to aboid value changing though SVD Decomposion
	float* mtxA_cpy_d = NULL;

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
    signed char jobU = 'S';

    //Specifies options for computing all or part of the matrix V**T: = ‘A’: 
    //all N rows of V**T are returned in the array
    signed char jobVT = 'S';

	//Error cheking after performing SVD decomp
	int infoGpu = 0;


	//(1) Allocate memoery, and copy value
	CHECK(cudaMalloc((void**)&mtxA_cpy_d, numOfRow * numOfClm * sizeof(float))); 
	CHECK(cudaMemcpy(mtxA_cpy_d, mtxA_d, numOfRow * numOfClm * sizeof(float), cudaMemcpyDeviceToDevice));

	CHECK((cudaMalloc((void**)&devInfo, sizeof(int))));

	//(2) Calculate workspace for SVD decompositoin
	checkCudaErrors(cusolverDnSgesvd_bufferSize(cusolverHandler, numOfRow, numOfClm, &lwork));
    CHECK(cudaMalloc((void**)&work_d, lwork * sizeof(float)));


    //(3) Compute SVD decomposition
    checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, numOfRow, numOfClm, mtxA_cpy_d, ldngDim, sngVals_d, mtxU_d,ldngDim, mtxVT_d, numOfClm, work_d, lwork, rwork_d, devInfo));
	
	//(4) Check SVD decomp was successful. 
	checkCudaErrors(cudaMemcpy(&infoGpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if(infoGpu != 0){
		printf("\n\n😖😖😖Unsuccessful SVD execution😖😖😖\n");
	}

	//(5) Free memoery
	checkCudaErrors(cudaFree(work_d));
	checkCudaErrors(cudaFree(devInfo));
	checkCudaErrors(cudaFree(mtxA_cpy_d));

	return;
}


//Input: singluar values, int currnet rank, float threashould
//Process: Check eigenvalues are greater than threashould, then set new rank
//Output: int newRank
int setRank(float* sngVals_d, int currentRank, float threashold)
{	
	int newRank = 0;

	//Allcoate in heap to copy value from device
	float* sngVals_h = (float*)malloc(currentRank * sizeof(float));
	// Copy singular values from Device to count eigen values
	checkCudaErrors(cudaMemcpy(sngVals_h, sngVals_d, currentRank * sizeof(float), cudaMemcpyDeviceToHost));

	for(int wkr = 0; wkr < currentRank; wkr++){
		if(sngVals_h[wkr] > threashold){
			newRank++;
		} // end of if
	} // end of for

	free(sngVals_h);

	return newRank;
}

//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, float* matrix A in device, float* matrix B in device , float* result matrix C in device, int leading dimension A, in leading dimension B
//Process: matrix multiplication matrix A and matrix B
//Result: matrix C as a result
void multiply_Den_ClmM_mtx_mtx(cublasHandle_t cublasHandler, float* mtxA_d, float* mtxB_d, float* mtxC_d, int numOfRowA, int numOfColB, int numOfColA)
{	
	const float alpha = 1.0f;
	const float beta = 0.0f;

	checkCudaErrors(cublasSgemm(cublasHandler, CUBLAS_OP_N, CUBLAS_OP_N, numOfRowA, numOfColB, numOfColA, &alpha, mtxA_d, numOfRowA, mtxB_d, numOfColA, &beta, mtxC_d, numOfRowA));

}


//Input matrix should be column major
//Input: cubasHandle_t cublasHandler, float* matrix A in device, float* result matrix C in device, int number of Row, int number of column
//Process: matrix multiplication matrix A' * matrix A
//Result: matrix C as a result with square matrix
void multiply_Den_ClmM_mtxT_mtx(cublasHandle_t cublasHandler, float* mtxA_d, float* mtxC_d, int numOfRow, int numOfClm)
{	
	const float alpha = 1.0f;
	const float beta = 0.0f;
	checkCudaErrors(cublasSgemm(cublasHandler, CUBLAS_OP_T, CUBLAS_OP_N, numOfClm, numOfClm, numOfRow, &alpha, mtxA_d, numOfRow, mtxA_d, numOfRow, &beta, mtxC_d, numOfClm));
	
	// It will be no longer need inside orth(*), and later iteration
	CHECK(cudaFree(mtxA_d)); 
} // end of multiply_Den_ClmM_mtxT_mtx



//Input matrix should be column major, overload function
//Input: cubasHandle_t cublasHandler, float* matrix A in device, float* matrix B in device , float* result matrix C in device, int number of Row A, int number of column A, int number of cloumn B
//Process: matrix multiplication matrix A' * matrix A
//Result: matrix C as a result with square matrix
void multiply_Den_ClmM_mtxT_mtx(cublasHandle_t cublasHandler, float* mtxA_d, float* mtxB_d, float* mtxC_d, int numOfRowA, int numOfClmA, int numOfClmB)
{
	
	const float alpha = 1.0f;
	const float beta = 0.0f;

	/*
	Note
	3rd parameter, m: Number of rows of C (or AT), which is numOfColA (K).
	4th parameter, n: Number of columns of C (or B), which is numOfColB (N).
	5th parameter, k: Number of columns of AT (or rows of B), which is numOfRowA (M).
	
	Summary,
	Thinking about 3rd and 4th parameter as matrix C would be colmnA * columnB because A is transposed.
	Then, the 5th parameter is inbetween number as rowA or rowB becuase matrix A is transpose
	*/
	checkCudaErrors(cublasSgemm(cublasHandler, CUBLAS_OP_T, CUBLAS_OP_N, numOfClmA, numOfClmB, numOfRowA, &alpha, mtxA_d, numOfRowA, mtxB_d, numOfRowA, &beta, mtxC_d, numOfClmA));

} // end of multiply_Den_ClmM_mtxT_mtx




//TO DO check this function is need or not.
//Input: cublasHandler_t cublasHandler, float* matrix X, int number of row, int number of column
//Process: the function allocate new memory space and tranpose the mtarix X
//Output: float* matrix X transpose
float* transpose_Den_Mtx(cublasHandle_t cublasHandler, float* mtxX_d, int numOfRow, int numOfClm)
{	
	float* mtxXT_d = NULL;
	const float alpha = 1.0f;
	const float beta = 0.0f;

	//Allocate a new memory space for mtxXT
	CHECK(cudaMalloc((void**)&mtxXT_d, numOfRow * numOfClm * sizeof(float)));
	
	//Transpose mtxX
	// checkCudaErrors(cublasSgeam(cublasHandler, CUBLAS_OP_T, CUBLAS_OP_N, COL_A, COL_A, &alpha, mtxVT_d, COL_A, &beta, mtxVT_d, COL_A, mtxV_d, COL_A));
    checkCudaErrors(cublasSgeam(cublasHandler, CUBLAS_OP_T, CUBLAS_OP_N, numOfRow, numOfClm, &alpha, mtxX_d, numOfClm, &beta, mtxX_d, numOfRow, mtxXT_d, numOfRow));

	//Free memory the original matrix X
	CHECK(cudaFree(mtxX_d));

	return mtxXT_d;
}

//Input: float* matrix V, int number of Row and Column, int current rank
//Process: the functions truncates the matrix V with current rank
//Output: float* matrix V truncated.
float* truncate_Den_Mtx(float* mtxV_d, int numOfN, int currentRank)
{	
	//Allocate memoery for truncated matrix V
	float* mtxV_trnc_d = NULL;
	CHECK(cudaMalloc((void**)&mtxV_trnc_d, numOfN * currentRank * sizeof(float)));

	//Copy value from the original matrix until valid column vectors
	CHECK(cudaMemcpy(mtxV_trnc_d, mtxV_d, numOfN * currentRank * sizeof(float), cudaMemcpyDeviceToDevice));

	//Make sure memoery Free full matrix V.
	CHECK(cudaFree(mtxV_d));

	//Return truncated matrix V.
	return mtxV_trnc_d;
}




//Input: float* mtxY, product of matrix Z * matrix U, int number of row, int number of column 
//Process: the kernel normalize each column vector of matrix Y in 2 norm
//Output: float* mtxY_d, which will be updated as normalized matrix Y hat.
 __global__ void normalizeClmVec(float* mtxY_d, int numOfRow, int numOfCol)
 {
	// TO DO optimize with shared memory

	//Calcualte global memory trhead ID
	int glbCol = blockIdx.x * blockDim.x + threadIdx.x;
	//Set boundry condition
	if(glbCol < numOfCol){
		// printf("glbCol %d\n", glbCol);
		//Calculate the L2 norm of the column
		float sqrSum = 0.0f;

		//sum of (column value)^2
		for (int wkr = 0; wkr < numOfRow; wkr++){
			// printf("mtxY_d[%d] =  %f\n", wkr,  mtxY_d [glbCol * numOfRow + wkr]);
			sqrSum += mtxY_d [glbCol * numOfRow + wkr] * mtxY_d [glbCol * numOfRow + wkr];
		}
		// printf("sqrSum %f\n", sqrSum);

		// scalar = 1 / √sum of (column value)^2  
		float normScaler = 1.0f /  sqrtf(sqrSum);

		// printf("normScaler %f\n", normScaler);

		//Normalize column value	
		if(normScaler > 0.0f){
			for (int wkr = 0; wkr < numOfRow; wkr++){
				float nrmVal = mtxY_d[ glbCol * numOfRow + wkr] * normScaler;
				// printf("nrmVal %f\n", nrmVal);
				mtxY_d[ glbCol * numOfRow + wkr] = nrmVal;
			} // enf of for
		} // end of if normlize column vector 
	} // end of if boundry condition

 } // end of normalizeClmVec


//Input: float* mtxY, product of matrix Z * matrix U, int number of row, int number of column 
//Process: the function calls kernel and normalize each column vector 
//Output: float* mtxY_d, which will be updated as normalized matrix Y hat.
void normalize_Den_Mtx(float* mtxY_d, int numOfRow, int numOfCol)
{		

	// Use a 1D block and grid configuration
    int blockSize = 1024; // Number of threads per block
    int gridSize = ceil((float)numOfCol / blockSize); // Number of blocks needed

    normalizeClmVec<<<gridSize, blockSize>>>(mtxY_d, numOfRow, numOfCol);
    
	cudaDeviceSynchronize(); // Ensure the kernel execution completes before proceeding
}


//Input: float* matrix A, int number of Row, int number Of Column
//Process: Compute condition number and check whther it is ill-conditioned or not.
//Output: float condition number
float computeConditionNumber(float* mtxA_d, int numOfRow, int numOfClm)
{
	bool debug = false;

	//Create handler
    cusolverDnHandle_t cusolverHandler = NULL;
    checkCudaErrors(cusolverDnCreate(&cusolverHandler));

	float* sngVals_d = extractSngVals(cusolverHandler, numOfRow, numOfClm, numOfRow, mtxA_d);
	if(debug){
		printf("\n\nsngular values after SVD decomp\n\n");
		print_vector(sngVals_d, numOfClm);
	}

	float* sngVals_h = (float*)malloc(numOfClm * sizeof(float));
	CHECK(cudaMemcpy(sngVals_h, sngVals_d, numOfClm * sizeof(float), cudaMemcpyDeviceToHost));
	float conditionNum = sngVals_h[0] / sngVals_h[numOfClm-1];
	
	checkCudaErrors(cusolverDnDestroy(cusolverHandler));
	CHECK(cudaFree(sngVals_d));
	free(sngVals_h);
	
	return conditionNum;

} // end of computeConditionNumber



//Input: cusolverDnHandle_t cusolverHandler, int number of row, int number of column, int leading dimension, float* matrix A
//Process: Extract eigenvalues with full SVD
//Output: float* sngVals_d, singular values in device in column vector
float* extractSngVals(cusolverDnHandle_t cusolverHandler, int numOfRow, int numOfClm, int ldngDim, float* mtxA_d)
{
	
	float *mtxA_cpy_d = NULL; // Need a copy to tranpose mtxZ'

	float *mtxU_d = NULL;
	float *sngVals_d = NULL;
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

	//Error cheking after performing SVD decomp
	int infoGpu = 0;

	bool debug = false;


	if(debug){
		printf("\n\n~~mtxA~~\n\n");
		print_mtx_clm_d(mtxA_d, numOfRow, numOfClm);
	}


	//(1) Allocate memeory in device
	//Make a copy of mtxZ for mtxZ'
    CHECK(cudaMalloc((void**)&mtxA_cpy_d, numOfRow * numOfClm * sizeof(float)));

	//For SVD decomposition
	CHECK(cudaMalloc((void**)&mtxU_d, numOfRow * numOfClm * sizeof(float)));
	CHECK(cudaMalloc((void**)&sngVals_d, numOfClm * sizeof(float)));
	CHECK(cudaMalloc((void**)&mtxVT_d, numOfClm * numOfClm * sizeof(float)));

	//(2) Copy value to device
	CHECK(cudaMemcpy(mtxA_cpy_d, mtxA_d, numOfRow * numOfClm * sizeof(float), cudaMemcpyDeviceToDevice));
	
	
	if(debug){
		printf("\n\n~~mtxA cpy~~\n\n");
		print_mtx_clm_d(mtxA_cpy_d, numOfRow, numOfClm);
	}


	//(4) Calculate workspace for SVD decompositoin
	checkCudaErrors(cusolverDnSgesvd_bufferSize(cusolverHandler, numOfRow, numOfClm, &lwork));
    CHECK(cudaMalloc((void**)&work_d, lwork * sizeof(float)));
	CHECK((cudaMalloc((void**)&devInfo, sizeof(int))));

    //(3) Compute SVD decomposition
    checkCudaErrors(cusolverDnSgesvd(cusolverHandler, jobU, jobVT, numOfRow, numOfClm, mtxA_cpy_d, ldngDim, sngVals_d, mtxU_d,ldngDim, mtxVT_d, numOfClm, work_d, lwork, rwork_d, devInfo));
	
	//(4) Check SVD decomp was successful. 
	checkCudaErrors(cudaMemcpy(&infoGpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	if(infoGpu != 0){
		printf("\n\n😖😖😖Unsuccessful SVD execution😖😖😖\n");
	}

	// if(debug){
	// 	printf("\n\n~~sngVals_d~~\n\n");
	// 	print_mtx_clm_d(sngVals_d, numOfClm, numOfClm);
	// }

	//(5) Free memoery
	checkCudaErrors(cudaFree(work_d));
	checkCudaErrors(cudaFree(devInfo));
	checkCudaErrors(cudaFree(mtxA_cpy_d));
	checkCudaErrors(cudaFree(mtxU_d));
	checkCudaErrors(cudaFree(mtxVT_d));


	return sngVals_d;

} // end of extractSngVals




#endif // HELPER_FUNCTIONS_H