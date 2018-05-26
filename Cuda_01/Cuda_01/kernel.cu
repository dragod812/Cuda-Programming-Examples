
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <conio.h>
#include <stdio.h>
#include <stdlib.h>
__global__ void add(int a, int b, int *c){
	*c = a + b;
}

__global__ void addMatrix(int *c, int *a, int *b){
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	*(c + blockDim.y*i + j) = *(a + blockDim.y*i + j) + *(b + blockDim.y*i + j);
}
cudaError_t addMatrixWC(int *C, int *A, int *B, int N);
/*
//addition of two numbers
int main()
{
int c;
int *dev_c;
cudaError_t cudaStatus;
cudaStatus = cudaMalloc((void**)&dev_c, sizeof(int));
if (cudaStatus != cudaSuccess){
fprintf(stderr, "cudaMalloc Failed!");
}
add << <1, 1 >> >(2, 7, dev_c);
cudaStatus = cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
if (cudaStatus != cudaSuccess){
fprintf(stderr, "cudaMemcpy Failed");
}
printf("ans %d", c);
getch();
cudaFree(dev_c);
return 0;
}
*/
int main()
{   
	int N;
	scanf("%d", &N);
	int *A = (int *)malloc(N*N*sizeof(int));
	int *B = (int *)malloc(N*N*sizeof(int));
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			scanf("%d", (A + N*i + j));
		}
	}
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			scanf("%d", (B + N*i + j));
		}
	}


	int *C = (int *)malloc(N*N*sizeof(int));
	cudaError_t cudaStatus = addMatrixWC(C, A, B, N);
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			printf("%d ", *(C + N*i + j));
		}
		printf("\n");
	}
	getch();
    return 0;
}
cudaError_t addMatrixWC(int *C, int *A, int *B, int N){
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	
	cudaError_t cudaStatus;
	printf("Inside Addmatrix!\n A:\n");
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			printf("%d ", *(A + N*i + j));
		}
		printf("\n");
	}
	printf("B:\n");
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			printf("%d ", *(B + N*i + j));
		}
		printf("\n");
	}
	cudaStatus = cudaMalloc((void**)&dev_a, N*N*sizeof(int));
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMalloc Failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_b, N*N*sizeof(int));
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMalloc Failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&dev_c, N*N*sizeof(int));
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMalloc Failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_a, A, N*N*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	cudaStatus = cudaMemcpy(dev_b, B, N*N*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	dim3 blockspergrid(N / 4, N / 4, 1);
	dim3 threadsperblock(4, 4, 1);
	addMatrix<<< blockspergrid, threadsperblock >> >(dev_c, dev_a, dev_b);
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
	
	cudaStatus = cudaMemcpy(C, dev_c, N*N*sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess){
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


Error:
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return cudaStatus;
}
// Helper function for using CUDA to add vectors in parallel.
/*
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
*/
