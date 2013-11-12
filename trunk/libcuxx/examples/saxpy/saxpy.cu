__noinline__ __device__ void saxpy(int* R, int* A, int* X, int b, int size)
{
	for(int i = 0; i < size; ++i)
	{
		R[i] = A[i] * X[i] + b;
	}
}

__global__ void saxpy_kernel(int* D, int* A, int* X, int b, int size)
{
	int threads = blockDim.x;
	int id      = threadIdx.x;

	for(int i = id; i < size; i+= threads)
	{
		D[i] = A[i] * X[i] + b;
	}
}

#include <cstdio>

// Allocated out here due to lack of correct local memory
__device__ int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
__device__ int X[] = {1, 2, 3, 5, 6, 7, 9, 4, 8};

__device__ int R[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
__device__ int D[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

__global__ void _main(int argc, char** argv)
{
	int size = sizeof(A);
	int b = 2;
	
	saxpy(R, A, X, b, size);
	
	saxpy_kernel<<<1, size>>>(D, A, X, b, size);
	cudaDeviceSynchronize();
	
	for(int i = 0; i < size; ++i)
	{
		std::printf("R[%d] = %d\n", i, R[i]);
		std::printf("D[%d] = %d\n", i, D[i]);
	}
}

