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
    
    float mtxA_h[] = {        
        10.840188, 0.394383, 0.000000, 0.000000, 0.000000,  
        0.394383, 10.783099, 0.798440, 0.000000, 0.000000, 
        0.000000, 0.798440, 10.911648, 0.197551, 0.000000, 
        0.000000, 0.000000, 0.197551, 10.335223, 0.768230, 
        0.000000, 0.000000, 0.000000, 0.768230, 10.277775 
    };

    float mtxSolX_h[] = {0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0};

    float mtxB_h[] = {1.0, 1.0, 1.0, 1.0, 1.0, 
                    1.0, 1.0, 1.0, 1.0, 1.0,
                    1.0, 1.0, 1.0, 1.0, 1.0};

    bool debug = true;

    //(1) Allocate memory
    float* mtxA_d = NULL;
    float* mtxSolX_d = NULL;
    float* mtxB_d = NULL;

    CHECK(cudaMalloc((void**)&mtxA_d,  M * K * sizeof(float)));
    CHECK(cudaMalloc((void**)&mtxSolX_d,  K * N * sizeof(float)));
    CHECK(cudaMalloc((void**)&mtxB_d,  M * N * sizeof(float)));

    //(2) Copy value from host to device
    CHECK(cudaMemcpy(mtxA_d, mtxA_h, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxSolX_d, mtxSolX_h, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(mtxB_d, mtxB_h, M * N * sizeof(float), cudaMemcpyHostToDevice));

    if(debug){
        printf("\n\n~~mtxA~~\n\n");
        print_mtx_clm_d(mtxA_d, M, K);
        printf("\n\n~~mtxSolX~~\n\n");
        print_mtx_clm_d(mtxSolX_d, K, N);
        printf("\n\n~~mtxB~~\n\n");
        print_mtx_clm_d(mtxB_d, M, N);
    }

    //Solve AX = B with bfbcg method
    bfbcg(mtxA_d, mtxSolX_d, mtxB_d, M, N);



    //()Free memeory
    CHECK(cudaFree(mtxA_d));
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