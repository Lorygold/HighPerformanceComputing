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
#define BLOCK_SIZE (32)
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
      A[i * nk + j] = (i * j) / (double)ni;
    }
  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i * nj + j] = (i * (j + 1)) / (double)nj;
  for (i = 0; i < nl; i++)
    for (j = 0; j < nj; j++)
      C[i * nj + j] = (i * (j + 3)) / (double)nl;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      D[i * nl + j] = (i * (j + 2)) / (double)nk;
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
  // Check bounds
  if (i < ni && j < nj) {
    // Initialize result
    double sum = 0.0;
    // Loop over k
    for (int k = 0; k < nk; k++){
      // Accumulate product
      sum += alpha * A[(i * nk) + k] * B[(k * nj) + j];
    }
    // Write result
    tmp[(i * nj) + j] = sum;
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
   for (int k = 0; k < nl; k++){
      // Accumulate product
      sum += tmp[(i * nl) + k] * C[(k * nl) + j];
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
  double* A = new double[ni*nk];
  double* B = new double[nk*nj];
  double* C = new double[nl*nj];
  double* D = new double[ni*nl];
  double* tmp = new double[ni*nj];

  /* Initialize array(s). */
  init_array(ni, nj, nk, nl, &alpha, &beta,A,B,C,D);

  /* Start timer. */
  polybench_start_instruments;

  // Allocate device memory
  double *d_a, *d_b, *d_c, *d_d, *d_tmp;
  gpuErrchk(cudaMalloc((void **)&d_a, sizeof(double) * ni * nk));
  gpuErrchk(cudaMalloc((void **)&d_b, sizeof(double) * nk * nj));
  gpuErrchk(cudaMalloc((void **)&d_c, sizeof(double) * nl * nj));
  gpuErrchk(cudaMalloc((void **)&d_d, sizeof(double) * ni * nl));
  gpuErrchk(cudaMalloc((void **)&d_tmp, sizeof(double) * ni * nj));
  //printf("N=%d, BLOCK_SIZE=%d grid_size=%d\n",N,BLOCK_SIZE,((N+(BLOCK_SIZE-1))/BLOCK_SIZE));
  // Data copy from host to device
  gpuErrchk(cudaMemcpy(d_a, A, sizeof(double) * ni * nk, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_b, B, sizeof(double) * nk * nj, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_c, C, sizeof(double) * nl * nj, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_d, D, sizeof(double) * ni * nl, cudaMemcpyHostToDevice));

  dim3 block_size(BLOCK_SIZE,BLOCK_SIZE);
  dim3 grid_size((N+BLOCK_SIZE-1) / (BLOCK_SIZE),(N+BLOCK_SIZE-1) / (BLOCK_SIZE));
  /* D := alpha*A*B*C + beta*D */
  printf("\ngrid_size=%d, block_size=%d\n",((N+BLOCK_SIZE-1) / (BLOCK_SIZE)) * ((N+BLOCK_SIZE-1) / (BLOCK_SIZE)), BLOCK_SIZE*BLOCK_SIZE );
  kernelCUDA1<<<grid_size,block_size>>>(d_tmp, d_a, d_b, alpha, ni, nj, nk);
  gpuErrchk( cudaPeekAtLastError() );
  gpuErrchk( cudaDeviceSynchronize() );
  kernelCUDA2<<<grid_size,block_size>>>(d_tmp, d_d, d_c, beta, ni, nj, nl);
  gpuErrchk(cudaMemcpy(tmp, d_tmp, sizeof(double) * ni * nj, cudaMemcpyDeviceToHost));
  for(int z = 0; z < ni*nj; z++){
    printf("value=%f\n",tmp[z]);
  }
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;


  /* Be clean. */
  delete[] tmp;
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] D;
  cudaFree(d_a);
  cudaFree(d_b);
  //cudaFree(d_c)
  //cudaFree(d_d)
  cudaFree(d_tmp);

  return 0;
}
