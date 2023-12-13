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
#define BLOCK_SIZE (1024)
#endif

/* Array initialization. */
static void init_array(int ni, int nj, int nk, int nl,
                       DATA_TYPE *alpha,  
                       DATA_TYPE *beta,
                       double* A,
                       double* B,
                       double* C,
                       double* D)
{
  int i, j;

  *alpha = 32412;
  *beta = 2123;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nk; j++)
      A[i * nk + j] = (i * j) / (double)ni;
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
    double sum = 0;
    // Loop over k
    for (int k = 0; k < nk; k++) {
      // Accumulate product
      sum += alpha * A[(i * nk) + k] * B[(k * nj) + j];
    }
    // Write result
    tmp[(i * nj) + j] = sum;
  }
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_2mm(int ni, int nj, int nk, int nl,
                       DATA_TYPE alpha,
                       DATA_TYPE beta,
                       DATA_TYPE POLYBENCH_2D(tmp, NI, NJ, ni, nj),
                       DATA_TYPE POLYBENCH_2D(A, NI, NK, ni, nk),
                       DATA_TYPE POLYBENCH_2D(B, NK, NJ, nk, nj),
                       DATA_TYPE POLYBENCH_2D(C, NL, NJ, nl, nj),
                       DATA_TYPE POLYBENCH_2D(D, NI, NL, ni, nl))
{
    //int i, j, k;
  
    // for (i = 0; i < _PB_NI; i++)
    //   for (j = 0; j < _PB_NL; j++)
    //   {
    //     D[i][j] *= beta;
    //     for (k = 0; k < _PB_NJ; ++k)
    //       D[i][j] += tmp[i][k] * C[k][j];
    //   }
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

  /*
  POLYBENCH_2D_ARRAY_DECL(tmp, DATA_TYPE, NI, NJ, ni, nj);
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NI, NK, ni, nk);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NK, NJ, nk, nj);
  POLYBENCH_2D_ARRAY_DECL(C, DATA_TYPE, NL, NJ, nl, nj);
  POLYBENCH_2D_ARRAY_DECL(D, DATA_TYPE, NI, NL, ni, nl);
  */
  /* Initialize array(s). */
  init_array(ni, nj, nk, nl, &alpha, &beta,A,B,C,D);

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  /*
  kernel_2mm(ni, nj, nk, nl,
             alpha, beta,
             POLYBENCH_ARRAY(tmp),
             POLYBENCH_ARRAY(A),
             POLYBENCH_ARRAY(B),
             POLYBENCH_ARRAY(C),
             POLYBENCH_ARRAY(D));
  */
  // Allocate device memory
  double *d_a, *d_b, *d_c, *d_d, *d_tmp;
  cudaMalloc((void **)&d_a, sizeof(double) * ni * nk);
  cudaMalloc((void **)&d_b, sizeof(double) * nk * NJ);
  //cudaMalloc((void **)&d_c, sizeof(DATA_TYPE) * NL * nj);
  //cudaMalloc((void **)&d_d, sizeof(DATA_TYPE) * ni * NL);
  cudaMalloc((void **)&d_tmp, sizeof(double) * ni * nj);
  // Data copy from host to device
  cudaMemcpy(d_a, A, sizeof(double) * ni * nk, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, sizeof(double) * nk * nj, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_c, C, sizeof(DATA_TYPE) * NL * nj, cudaMemcpyHostToDevice);
  //cudaMemcpy(d_d, D, sizeof(DATA_TYPE) * ni * NL, cudaMemcpyHostToDevice);
  dim3 grid_size((N+(BLOCK_SIZE-1))/BLOCK_SIZE);
  dim3 block_size(BLOCK_SIZE);
  /* D := alpha*A*B*C + beta*D */
  kernelCUDA1<<<grid_size,block_size>>>(tmp, A, B, alpha, ni, nj, nk);
  cudaMemcpy(tmp, d_tmp, sizeof(double) * ni * nj, cudaMemcpyDeviceToHost);
  fprintf(stderr, "YOOO\n");
  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  //polybench_prevent_dce(print_array(ni, nl, POLYBENCH_ARRAY(D)));

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
