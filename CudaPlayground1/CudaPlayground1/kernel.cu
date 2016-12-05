
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <memory.h>
#include <memory>
#include <algorithm>
#include <iostream>

struct free_delete
{
	void operator()(void* x) { free(x); }
};

static void FillVector(int* v, size_t numberOfElements);
static bool CheckResult(int *c, const int *a, const int *b, unsigned int size);
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void addKernel2(int numberOfElements, int elementsPerInvocation, int *c, const int *a, const int *b)
{
	for (int i = 0; i < elementsPerInvocation; ++i)
	{
		int idx = blockIdx.x * blockDim.x * elementsPerInvocation + threadIdx.x*elementsPerInvocation;
		if (idx < numberOfElements)
		{
			c[idx+i] = a[idx+i] + b[idx+i];
		}
	}
}

int main()
{
	//const int arraySize = 5;
	//const int a[arraySize] = { 1, 2, 3, 4, 5 };
	//const int b[arraySize] = { 10, 20, 30, 40, 50 };
	//int c[arraySize] = { 0 };

	//// Add vectors in parallel.
	//cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	//if (cudaStatus != cudaSuccess) {
	//    fprintf(stderr, "addWithCuda failed!");
	//    return 1;
	//}

	//printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
	//    c[0], c[1], c[2], c[3], c[4]);

	const size_t ArraySize = 1024 * 1024 * 256;
	std::unique_ptr<int, free_delete> a((int*)malloc(ArraySize * sizeof(int)));
	FillVector(a.get(), ArraySize);
	std::unique_ptr<int, free_delete> b((int*)malloc(ArraySize * sizeof(int)));
	FillVector(b.get(), ArraySize);
	std::unique_ptr<int, free_delete> c((int*)malloc(ArraySize * sizeof(int)));

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c.get(), a.get(), b.get(), ArraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	bool isCorrect = CheckResult(c.get(), a.get(), b.get(), ArraySize);
	if (isCorrect)
	{
		fprintf(stdout, "Result is correct!");
	}
	else
	{
		fprintf(stdout, "Result is NOT correct!");
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

bool CheckResult(int *c, const int *a, const int *b, unsigned int size)
{
	for (auto i = 0; i < size; ++i)
	{
		if (c[i] != a[i] + b[i])
			return false;
	}

	return true;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;

	int blockSize;
	int n_blocks;
	int elementsPerThread;

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

	/*fprintf(stdout, "Memory allocated, press any key!");
	std::cin.ignore();
	fprintf(stdout, "\n");*/
	
	blockSize = 512;
	n_blocks = (std::min)((int)(size / blockSize + (size%blockSize == 0 ? 0 : 1)), 1024);
	elementsPerThread = size / (blockSize*n_blocks);

	addKernel2 << <n_blocks, blockSize >> > (size, (std::max)(elementsPerThread, 1), dev_c, dev_a, dev_b);

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

void FillVector(int* v, size_t numberOfElements)
{
	for (size_t i = 0; i < numberOfElements; ++i)
	{
		*(v + i) = (int)i;
	}
}