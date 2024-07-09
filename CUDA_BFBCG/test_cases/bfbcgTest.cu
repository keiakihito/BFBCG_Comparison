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


void bfbcgTest_Case1();
void bfbcgTest_Case2();
void bfbcgTest_Case3();
void bfbcgTest_Case4();
void bfbcgTest_Case5();

int main(int arg, char** argv)
{
    
    printf("\n\n= = = =bfbcgTest.cu= = = = \n\n");
    
    printf("\n\n🔍🔍🔍 Test Case 1 🔍🔍🔍\n\n");
    bfbcgTest_Case1();

    // printf("\n\n🔍🔍🔍 Test Case 2 🔍🔍🔍\n\n");
    // bfbcgTest_Case2();

    // printf("\n\n🔍🔍🔍 Test Case 3 🔍🔍🔍\n\n");
    // bfbcgTest_Case3();

    // printf("\n\n🔍🔍🔍 Test Case 4 🔍🔍🔍\n\n");
    // bfbcgTest_Case4();

    // printf("\n\n🔍🔍🔍 Test Case 5 🔍🔍🔍\n\n");
    // bfbcgTest_Case5();

    printf("\n\n= = = = end of bfbcgTest = = = =\n\n");

    return 0;
} // end of main




void bfbcgTest_Case1()
{
    const int M = 5;
    const int K = 5;
    const int N = 3;
    
    // float mtxA_h[] = {        
    //     10.840188, 0.394383, 0.000000, 0.000000, 0.000000,  
    //     0.394383, 10.783099, 0.798440, 0.000000, 0.000000, 
    //     0.000000, 0.798440, 10.911648, 0.197551, 0.000000, 
    //     0.000000, 0.000000, 0.197551, 10.335223, 0.768230, 
    //     0.000000, 0.000000, 0.000000, 0.768230, 10.277775 
    // };

    //Sparse matrix 
    
    int rowOffSets[] = {0, 2, 5, 8, 11, 13};
    int colIndices[] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4};
    float vals[] = {10.840188, 0.394383, 
                      0.394383, 10.783099, 0.798440, 
                      0.798440, 10.911648, 0.197551,
                      0.197551, 10.335223, 0.768230,
                      0.768230, 10.277775};

    int nnz = 13;

    //Generate sparse Identity matrix
    // CSRMatrix csrMtxA_h;

    // csrMtxA_h.numOfRows = M;
    // csrMtxA_h.numOfClms = K;
    // csrMtxA_h.numOfnz = nnz;
    // csrMtxA_h.row_offsets =  (int*)malloc((M + 1) * sizeof(int));
    // csrMtxA_h.col_indices = (int*)malloc(nnz * sizeof(int));
    // csrMtxA_h.vals = (float*)malloc(nnz * sizeof(float));

    // memcpy(csrMtxA_h.row_offsets, rowOffSets, (M+1) * sizeof(int));
    // memcpy(csrMtxA_h.col_indices, colIndices, nnz * sizeof(int));
    // memcpy(csrMtxA_h.vals, vals, nnz * sizeof(float));

    CSRMatrix csrMtxA_h = constructCSRMatrix(M, K, nnz, rowOffSets, colIndices, vals);


    float mtxSolX_h[] = {0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0};

    float mtxB_h[] = {1.0, 1.0, 1.0, 1.0, 1.0, 
                    1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0};

    bool debug = true;

    //(1) Allocate memory
    // float* mtxA_d = NULL;
    float* mtxSolX_d = NULL;
    float* mtxB_d = NULL;

    // CHECK(cudaMalloc((void**)&mtxA_d,  M * K * sizeof(float)));
    CHECK(cudaMalloc((void**)&mtxSolX_d,  K * N * sizeof(float)));
    CHECK(cudaMalloc((void**)&mtxB_d,  M * N * sizeof(float)));

    //(2) Copy value from host to device
    // CHECK(cudaMemcpy(mtxA_d, mtxA_h, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxSolX_d, mtxSolX_h, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxB_d, mtxB_h, M * N * sizeof(float), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~csrMtxA_h~~\n\n");
        print_CSRMtx(csrMtxA_h);
        printf("\n\n~~mtxSolX~~\n\n");
        print_mtx_clm_d(mtxSolX_d, K, N);
        printf("\n\n~~mtxB~~\n\n");
        print_mtx_clm_d(mtxB_d, M, N);
    }

    //Solve AX = B with bfbcg method
    bfbcg(csrMtxA_h, mtxSolX_d, mtxB_d, M, N);

    if(debug){
        printf("\n\n~~📝📝📝Approximate Solution Marix📝📝📝~~\n\n");
        print_mtx_clm_d(mtxSolX_d, K, N);
    }

    //Validate
    //R = B - AX
    printf("\n\n~~🔍👀Validate Solution Matrix X 🔍👀~~");
    float twoNorms = validateBFBCG(csrMtxA_h, M, mtxSolX_d, N, mtxB_d);
    printf("\n\n= = 1st Column Vector 2 norms: %f = =\n\n", twoNorms);


    //()Free memeory
    freeCSRMatrix(csrMtxA_h);
    CHECK(cudaFree(mtxSolX_d));
    CHECK(cudaFree(mtxB_d));



} // end of tranposeTest_Case1




void bfbcgTest_Case2()
{


} // end of tranposeTest_Case2




void bfbcgTest_Case3()
{
  
} // end of tranposeTest_Case3




void bfbcgTest_Case4()
{

} // end of tranposeTest_Case4




void bfbcgTest_Case5()
{
    
}// end of tranposeTest_Case4

/*
Sample Run



= = = =bfbcgTest.cu= = = = 



🔍🔍🔍 Test Case 1 🔍🔍🔍



~~csrMtxA_h~~



numOfRows: 5, numOfClms: 5 , number of non zero: 13

row_offsets: 
[ 0 2 5 8 11 13 ]


col_indices: 
[ 0 1 0 1 2 1 2 3 2 3 4 3 4 ]


non zero values: 
[ 10.840188 0.394383 0.394383 10.783099 0.798440 0.798440 10.911648 0.197551 0.197551 10.335223 0.768230 0.768230 10.277775 ]



~~mtxSolX~~

0.000000 0.000000 0.000000 
0.000000 0.000000 0.000000 
0.000000 0.000000 0.000000 
0.000000 0.000000 0.000000 
0.000000 0.000000 0.000000 


~~mtxB~~

1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 


~~mtxR~~

1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 


~~mtxR~~

1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 


~~original residual: 2.236068~~



~~mtxZ~~

1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 


~~mtxR~~

1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 


 = =  Current Rank: 1 = = 



~~mtxP~~

-0.447214 
-0.447214 
-0.447214 
-0.447214 
-0.447214 


💫💫💫 Iteration 1 💫💫💫



~~mtxQ~~

-5.024253 
-5.355795 
-5.325258 
-5.053963 
-4.939924 


~~mtxPTQ~~

11.493028 


~~mtxPTQ_inv~~

0.087009 


~~mtxPTR~~

-2.236068 -2.236068 -2.236068 


 = = mtxPTQ_inv_d: 0.087009 = = 


~~mtxAlph_d~~

-0.194559 -0.194559 -0.194559 


~~mtxSolX_d~~

0.087009 0.087009 0.087009 
0.087009 0.087009 0.087009 
0.087009 0.087009 0.087009 
0.087009 0.087009 0.087009 
0.087009 0.087009 0.087009 


~~mtxR_d~~

0.022488 0.022488 0.022488 
-0.042016 -0.042016 -0.042016 
-0.036075 -0.036075 -0.036075 
0.016708 0.016708 0.016708 
0.038895 0.038895 0.038895 


🔍🔍🔍Relative Residue: 0.032755🔍🔍🔍



~~mtxZ~~

0.022488 0.022488 0.022488 
-0.042016 -0.042016 -0.042016 
-0.036075 -0.036075 -0.036075 
0.016708 0.016708 0.016708 
0.038895 0.038895 0.038895 


~~mtxR~~

0.022488 0.022488 0.022488 
-0.042016 -0.042016 -0.042016 
-0.036075 -0.036075 -0.036075 
0.016708 0.016708 0.016708 
0.038895 0.038895 0.038895 


~~mtxQTZ~~

0.027572 0.027572 0.027572 


~~mtxBta_d~~

-0.002399 -0.002399 -0.002399 


~~ (mtxZ_{i+1}_d + p * beta) ~~

0.023561 0.023561 0.023561 
-0.040943 -0.040943 -0.040943 
-0.035002 -0.035002 -0.035002 
0.017781 0.017781 0.017781 
0.039968 0.039968 0.039968 


~~ mtxP_d <- orth(*)~~

-0.321513 
0.558711 
0.477636 
-0.242635 
-0.545402 


= = current Rank: 1 = = 



💫💫💫 Iteration 2 💫💫💫



~~mtxQ~~

-3.264917 
6.279203 
5.609959 
-2.832328 
-5.791916 


~~mtxPTQ~~

11.083638 


~~mtxPTQ_inv~~

0.090223 


~~mtxPTR~~

-0.073203 -0.073203 -0.073203 


 = = mtxPTQ_inv_d: 0.090223 = = 


~~mtxAlph_d~~

-0.006605 -0.006605 -0.006605 


~~mtxSolX_d~~

0.089133 0.089133 0.089133 
0.083319 0.083319 0.083319 
0.083855 0.083855 0.083855 
0.088612 0.088612 0.088612 
0.090611 0.090611 0.090611 


~~mtxR_d~~

0.000925 0.000925 0.000925 
-0.000544 -0.000544 -0.000544 
0.000977 0.000977 0.000977 
-0.001999 -0.001999 -0.001999 
0.000642 0.000642 0.000642 


🔍🔍🔍Relative Residue: 0.001141🔍🔍🔍



~~mtxZ~~

0.000925 0.000925 0.000925 
-0.000544 -0.000544 -0.000544 
0.000977 0.000977 0.000977 
-0.001999 -0.001999 -0.001999 
0.000642 0.000642 0.000642 


~~mtxR~~

0.000925 0.000925 0.000925 
-0.000544 -0.000544 -0.000544 
0.000977 0.000977 0.000977 
-0.001999 -0.001999 -0.001999 
0.000642 0.000642 0.000642 


~~mtxQTZ~~

0.000986 0.000986 0.000986 


~~mtxBta_d~~

-0.000089 -0.000089 -0.000089 


~~ (mtxZ_{i+1}_d + p * beta) ~~

0.000953 0.000953 0.000953 
-0.000594 -0.000594 -0.000594 
0.000934 0.000934 0.000934 
-0.001977 -0.001977 -0.001977 
0.000690 0.000690 0.000690 


~~ mtxP_d <- orth(*)~~

-0.373337 
0.232694 
-0.365917 
0.774273 
-0.270330 


= = current Rank: 1 = = 



💫💫💫 Iteration 3 💫💫💫



~~mtxQ~~

-3.955269 
2.069762 
-3.654011 
7.722323 
-2.183575 


~~mtxPTQ~~

9.864809 


~~mtxPTQ_inv~~

0.101370 


~~mtxPTR~~

-0.002550 -0.002550 -0.002550 


 = = mtxPTQ_inv_d: 0.101370 = = 


~~mtxAlph_d~~

-0.000259 -0.000259 -0.000259 


~~mtxSolX_d~~

0.089229 0.089229 0.089229 
0.083259 0.083259 0.083259 
0.083949 0.083949 0.083949 
0.088412 0.088412 0.088412 
0.090681 0.090681 0.090681 


~~mtxR_d~~

-0.000098 -0.000098 -0.000098 
-0.000009 -0.000009 -0.000009 
0.000032 0.000032 0.000032 
-0.000002 -0.000002 -0.000002 
0.000077 0.000077 0.000077 


🔍🔍🔍Relative Residue: 0.000058🔍🔍🔍



~~mtxZ~~

-0.000098 -0.000098 -0.000098 
-0.000009 -0.000009 -0.000009 
0.000032 0.000032 0.000032 
-0.000002 -0.000002 -0.000002 
0.000077 0.000077 0.000077 


~~mtxR~~

-0.000098 -0.000098 -0.000098 
-0.000009 -0.000009 -0.000009 
0.000032 0.000032 0.000032 
-0.000002 -0.000002 -0.000002 
0.000077 0.000077 0.000077 


~~mtxQTZ~~

0.000064 0.000064 0.000064 


~~mtxBta_d~~

-0.000007 -0.000007 -0.000007 


~~ (mtxZ_{i+1}_d + p * beta) ~~

-0.000095 -0.000095 -0.000095 
-0.000011 -0.000011 -0.000011 
0.000035 0.000035 0.000035 
-0.000007 -0.000007 -0.000007 
0.000079 0.000079 0.000079 


~~ mtxP_d <- orth(*)~~








= = current Rank: 0 = = 



!!!Current Rank became 0!!!



~~📝📝📝Approximate Solution Marix📝📝📝~~

0.089229 0.089229 0.089229 
0.083259 0.083259 0.083259 
0.083949 0.083949 0.083949 
0.088412 0.088412 0.088412 
0.090681 0.090681 0.090681 


~~🔍👀Validate Solution Matrix X 🔍👀~~

mtxR = B - AX
~~mtxR~~

-0.000098 -0.000098 -0.000098 
-0.000009 -0.000009 -0.000009 
0.000032 0.000032 0.000032 
-0.000002 -0.000002 -0.000002 
0.000077 0.000077 0.000077 


= = Residual 2 norms: 0.000129 = =



= = = = end of bfbcgTest = = = =

[kkatsumi@gpub001 test_cases]$ nvcc bfbcgTest.cu -o bfbcgTest  -lcublas -lcusolver -lcusparse
[kkatsumi@gpub001 test_cases]$ ./bfbcgTest


= = = =bfbcgTest.cu= = = = 



🔍🔍🔍 Test Case 1 🔍🔍🔍



~~csrMtxA_h~~



numOfRows: 5, numOfClms: 5 , number of non zero: 13

row_offsets: 
[ 0 2 5 8 11 13 ]


col_indices: 
[ 0 1 0 1 2 1 2 3 2 3 4 3 4 ]


non zero values: 
[ 10.840188 0.394383 0.394383 10.783099 0.798440 0.798440 10.911648 0.197551 0.197551 10.335223 0.768230 0.768230 10.277775 ]



~~mtxSolX~~

0.000000 0.000000 0.000000 
0.000000 0.000000 0.000000 
0.000000 0.000000 0.000000 
0.000000 0.000000 0.000000 
0.000000 0.000000 0.000000 


~~mtxB~~

1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 



~~mtxR~~

1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 


~~original residual: 2.236068~~



~~mtxZ~~

1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 


~~mtxR~~

1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 
1.000000 1.000000 1.000000 


 = =  Current Rank: 1 = = 



~~mtxP~~

-0.447214 
-0.447214 
-0.447214 
-0.447214 
-0.447214 


💫💫💫 Iteration 1 💫💫💫



~~mtxQ~~

-5.024253 
-5.355795 
-5.325258 
-5.053963 
-4.939924 


~~mtxPTQ~~

11.493028 


~~mtxPTQ_inv~~

0.087009 


~~mtxPTR~~

-2.236068 -2.236068 -2.236068 


 = = mtxPTQ_inv_d: 0.087009 = = 


~~mtxAlph_d~~

-0.194559 -0.194559 -0.194559 


~~mtxSolX_d~~

0.087009 0.087009 0.087009 
0.087009 0.087009 0.087009 
0.087009 0.087009 0.087009 
0.087009 0.087009 0.087009 
0.087009 0.087009 0.087009 


~~mtxR_d~~

0.022488 0.022488 0.022488 
-0.042016 -0.042016 -0.042016 
-0.036075 -0.036075 -0.036075 
0.016708 0.016708 0.016708 
0.038895 0.038895 0.038895 


🔍🔍🔍Relative Residue: 0.032755🔍🔍🔍



~~mtxZ~~

0.022488 0.022488 0.022488 
-0.042016 -0.042016 -0.042016 
-0.036075 -0.036075 -0.036075 
0.016708 0.016708 0.016708 
0.038895 0.038895 0.038895 


~~mtxR~~

0.022488 0.022488 0.022488 
-0.042016 -0.042016 -0.042016 
-0.036075 -0.036075 -0.036075 
0.016708 0.016708 0.016708 
0.038895 0.038895 0.038895 


~~mtxQTZ~~

0.027572 0.027572 0.027572 


~~mtxBta_d~~

-0.002399 -0.002399 -0.002399 


~~ (mtxZ_{i+1}_d + p * beta) ~~

0.023561 0.023561 0.023561 
-0.040943 -0.040943 -0.040943 
-0.035002 -0.035002 -0.035002 
0.017781 0.017781 0.017781 
0.039968 0.039968 0.039968 


~~ mtxP_d <- orth(*)~~

-0.321513 
0.558711 
0.477636 
-0.242635 
-0.545402 


= = current Rank: 1 = = 



...

💫💫💫 Iteration 3 💫💫💫


...


~~mtxR_d~~

-0.000098 -0.000098 -0.000098 
-0.000009 -0.000009 -0.000009 
0.000032 0.000032 0.000032 
-0.000002 -0.000002 -0.000002 
0.000077 0.000077 0.000077 


🔍🔍🔍Relative Residue: 0.000058🔍🔍🔍



~~mtxZ~~

-0.000098 -0.000098 -0.000098 
-0.000009 -0.000009 -0.000009 
0.000032 0.000032 0.000032 
-0.000002 -0.000002 -0.000002 
0.000077 0.000077 0.000077 


~~mtxR~~

-0.000098 -0.000098 -0.000098 
-0.000009 -0.000009 -0.000009 
0.000032 0.000032 0.000032 
-0.000002 -0.000002 -0.000002 
0.000077 0.000077 0.000077 


~~mtxQTZ~~

0.000064 0.000064 0.000064 


~~mtxBta_d~~

-0.000007 -0.000007 -0.000007 


~~ (mtxZ_{i+1}_d + p * beta) ~~

-0.000095 -0.000095 -0.000095 
-0.000011 -0.000011 -0.000011 
0.000035 0.000035 0.000035 
-0.000007 -0.000007 -0.000007 
0.000079 0.000079 0.000079 


~~ mtxP_d <- orth(*)~~








= = current Rank: 0 = = 



!!!Current Rank became 0!!!
🔸Exit iteration🔸



~~📝📝📝Approximate Solution Marix📝📝📝~~

0.089229 0.089229 0.089229 
0.083259 0.083259 0.083259 
0.083949 0.083949 0.083949 
0.088412 0.088412 0.088412 
0.090681 0.090681 0.090681 


~~🔍👀Validate Solution Matrix X 🔍👀~~

mtxR = B - AX
~~mtxR~~

-0.000098 -0.000098 -0.000098 
-0.000009 -0.000009 -0.000009 
0.000032 0.000032 0.000032 
-0.000002 -0.000002 -0.000002 
0.000077 0.000077 0.000077 


= = 1st Column Vector 2 norms: 0.000129 = =



= = = = end of bfbcgTest = = = =