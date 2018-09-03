// system libraries
// use nvcc -o (output name) -Wno-deprecated-gpu-targets -std=c++11 -Xcompiler -fopenmp  file_name.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <chrono>

// size definition. modify as needed
#define N 2000

using namespace std;

// safe call definition
static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number){
	if(err!=cudaSuccess){
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

// safe call definition
#define SAFE_CALL(call,msg) _safe_cuda_call(call,msg,__FILE__,__LINE__)

// initialize major row matrix
void initializeMatrix(long *ip, const int nxy){
  for(int i = 0; i < nxy; i++)
      ip[i] = i;
    return;
}

// utility function to check result
void checkResult(long *hostRef, long *gpuRef, const int nxy){
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < nxy; i++){
        if (abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("host %ld gpu %ld\n", hostRef[i], gpuRef[i]);
            break;
        }
    }

    if (match)
        printf("Arrays match.\n\n");
    else
        printf("Arrays do not match.\n\n");
}

// multiply matrix on host
void multiplyMatrixOnHost(long *A, long *B, long *C, const int nxy){

    long *ia = A;
    long *ib = B;
    long *ic = C;

		for(int i = 0; i < nxy; i++) {
			 for(int j = 0; j < nxy; j++) {
					 for(int k = 0; k < nxy; k++) {
							 ic[i * nxy + j] += ia[i * nxy + k] * ib[j + k * nxy];
					 }
			 }
	 }

    return;
}

// function to multiply matrix on host with threads
void multiplyMatrixOnHostThreads(long *A, long *B, long *C, const int nxy){

    long *ia = A;
    long *ib = B;
    long *ic = C;

    int i = 0;
    // use the pragma directive to automatically paralelize
    #pragma omp parallel for private(i) shared(ia, ib, ic)
		for(i = 0; i < nxy; i++) {
			 for(int j = 0; j < nxy; j++) {
					 for(int k = 0; k < nxy; k++) {
							 ic[i * nxy + j] += ia[i * nxy + k] * ib[j + k * nxy];
					 }
			 }
	 }

    return;
}

// kernel to multiply matrix on gpu
__global__ void multiplyMatrixOnGPU(long *A, long *B, long *C, const int nxy){

		// get ix and iy from cuda defined variables
		unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x;
		unsigned int iy = blockIdx.y * blockDim.y + threadIdx.y;

    long *ia = A;
    long *ib = B;
    long *ic = C;

    if (ix < nxy && iy < nxy){
        for(int i = 0; i < nxy ; i++)
            ic[iy * nxy + ix] += ia[iy * nxy + i] * ib[i * nxy + ix];
    }

}

int main(int argc, char* argv[]) {
  printf("%s Starting...\n", argv[0]);

  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  SAFE_CALL(cudaSetDevice(dev), "Error setting device");

  int nx = N;
  int ny = N;
  int nxy = nx * ny;
  int nBytes = nxy * sizeof(long*);
  printf("Matrix size: nx %d ny %d\n", nx, ny);

  // malloc host memory
  long *h_A = (long *)malloc(nBytes);
  long *h_B = (long *)malloc(nBytes);
  long *hostRef = (long *)malloc(nBytes);
  long *hostRefThreads = (long *)malloc(nBytes);
  long *gpuRef = (long *)malloc(nBytes);

  // initialize matrix
  initializeMatrix(h_A, nxy);
  initializeMatrix(h_B, nxy);

  // initialize to 0
  memset(hostRef, 0, nBytes);
  memset(hostRefThreads, 0, nBytes);
  memset(gpuRef, 0, nBytes);

  // multiply matrix on host
  auto start_cpu = std::chrono::high_resolution_clock::now();
  multiplyMatrixOnHost(h_A, h_B, hostRef, nx);
  auto end_cpu =  std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
  printf("multiplyMatrixOnHost elapsed %f ms\n", duration_ms.count());

  // multiply matrix on host with threads
  start_cpu =  std::chrono::high_resolution_clock::now();
  multiplyMatrixOnHostThreads(h_A, h_B, hostRefThreads, nx);
  end_cpu =  std::chrono::high_resolution_clock::now();
  duration_ms = end_cpu - start_cpu;
  printf("multiplyMatrixOnHostThreads elapsed %f ms\n", duration_ms.count());

  // check results
  checkResult(hostRef, hostRefThreads, nx);

  // malloc device global memory
  long *d_MatA, *d_MatB, *d_MatC;
  SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
  SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
  SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

  // transfer data from host to device
  SAFE_CALL(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
  SAFE_CALL(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");
  SAFE_CALL(cudaMemset(d_MatC, 0, nBytes), "Error copying d_MatB");

  // kernel definition and launch
  dim3 block(512, 1);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  // launch
  start_cpu = std::chrono::high_resolution_clock::now();
  multiplyMatrixOnGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx);
  SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
  end_cpu =  std::chrono::high_resolution_clock::now();

  // measure total time
  duration_ms = end_cpu - start_cpu;
  printf("multiplyMatrixOnGPU elapsed %f ms\n", duration_ms.count());

  // SAFE_CALL kernel error
  SAFE_CALL(cudaGetLastError(), "Error with last error");

  // copy kernel result back to host side
  SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");

  // check device results
  checkResult(hostRef, gpuRef, nx);

  // free device global memory
  SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
  SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
  SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

  // free host memory
  free(h_A);
  free(h_B);
  free(hostRef);
  free(hostRefThreads);
  free(gpuRef);

  // reset device
  SAFE_CALL(cudaDeviceReset(), "Error reseting");

  return (0);

}
