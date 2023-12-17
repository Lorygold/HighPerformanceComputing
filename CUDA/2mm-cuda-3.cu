#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include CUDA */
#include <cuda_runtime.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "2mm.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE (16)
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


/* Array initialization. */
static void init_array(int ni, int nj, int nk, int nl,
                       double *alpha,  
                       double *beta,
                       double* A,
                       double* B,
                       double* C,
                       double* D)
{
  int i, j;
  //printf("ni=%d nk=%d nj=%d nl=%d\n", ni, nk, nj, nl);
  *alpha = 32412;
  *beta = 2123;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++){
      A[i * nk + j] = ((double)i * j) / ni;
    }
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i * nj + j] = ((double)i * (j + 1)) / nj;
  for (i = 0; i < nl; i++)
    for (j = 0; j < nj; j++)
      C[i * nj + j] = ((double)i * (j + 3)) / nl;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D[i * nl + j] = ((double)i * (j + 2)) / nk;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nl,
                        DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, D[i][j]);
      if ((i * ni + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}

__global__ void kernelCUDA1(double* tmp, double* A, double* B, double alpha, int ni, int nj, int nk)
{
  // Get row and column indices
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k;
  // Check bounds
  if (i < ni && j < nj) 
  {
    
    // Initialize result
    double sum = 0.0;
    // Loop over k
    for(k = 0; k < nk; k++)
    {  
      
      // Accumulate product
      sum = sum + alpha * B[(i * nk) + k] * A[(k * nj) + j];
      //printf("i=%d nk=%d k=%d j=%d nj=%d result_1=%d result_2=%d value_a=%f\n",i,nk,k,j,nj,((i * nk) + k),((k * nj) + j),A[(i * nk) + k]);
    }
    // Write result
    tmp[(i * nj) + j] = sum;
    //printf("%f\n",sum);
  }
}

__global__ void kernelCUDA2(double* tmp, double* D, double* C, double beta, int ni, int nj, int nl)
{

  // Get row and column indices
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  // Check bounds
  if (i < ni && j < nj) {
    
   double sum = 0.0; 
   for (int k = 0; k < nj; k++){
      // Accumulate product
      sum += tmp[(i * nj) + k] * C[(k * nj) + j];
    }
    // Write result
    D[(i * nl) + j] = sum*beta;
  }
}


int main(int argc, char **argv)
{
  
  int N = NI;
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;
  /* Variable declaration/allocation. */
  double alpha;
  double beta;
  double* A;
  double* B;
  double* C;
  double* D;
  double* tmp;

  /* Initialize array(s). */
  gpuErrchk(cudaMallocManaged((void **)&A, sizeof(double) * ni * nk));
  gpuErrchk(cudaMallocManaged((void **)&B, sizeof(double) * nk * nj));
  gpuErrchk(cudaMallocManaged((void **)&C, sizeof(double) * nl * nj));
  gpuErrchk(cudaMallocManaged((void **)&D, sizeof(double) * ni * nl));
  gpuErrchk(cudaMallocManaged((void **)&tmp, sizeof(double) * ni * nj));
  init_array(ni, nj, nk, nl, &alpha, &beta,A,B,C,D);
  
  
  /* Start timer. */
  polybench_start_instruments;

  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid_size((N+BLOCK_SIZE-1) / (BLOCK_SIZE),(N+BLOCK_SIZE-1) / (BLOCK_SIZE));
  printf("\ngrid_size=%d, block_size=%d\n",((N+BLOCK_SIZE-1) / (BLOCK_SIZE)) * ((N+BLOCK_SIZE-1) / (BLOCK_SIZE)), BLOCK_SIZE*BLOCK_SIZE );
  /* D := alpha*A*B*C + beta*D */
  kernelCUDA1<<<grid_size,block_size>>>(tmp, A, B, alpha, ni, nj, nk);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaPeekAtLastError());
  kernelCUDA2<<<grid_size,block_size>>>(tmp, D, C, beta, ni, nj, nl);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaPeekAtLastError());
  
  for(int z=0; z< ni*nk; z++){
    printf("z=%d D[z]=%f\n ", z,D[z]);
  }
  
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Be clean. */
  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(D);
  cudaFree(tmp);

  return 0;
}
