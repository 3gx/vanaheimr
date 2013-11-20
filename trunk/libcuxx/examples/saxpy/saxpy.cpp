#include <parallel>
#include <iostream>

void saxpy(int* R, int a, int* X, int b, int size)
{
	for(int i = 0; i < size; ++i)
	{
		R[i] = a * X[i] + b;
	}
}

void saxpy_kernel(const std::parallel_context& context,
	int* D, int a, int* X, int b, int size)
{
	int threads = context.total_threads();
	int id      = context.thread_id_in_context();

	for(int i = id; i < size; i+= threads)
	{
		D[i] = a * X[i] + b;
	}
}

int main(int argc, char** argv)
{
	int a   = 4;
	int X[] = {1, 2, 3, 5, 6, 7, 9, 4, 8};
	int b   = 2;
	
	size_t size = sizeof(X);
	
	int R[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	int D[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	
	saxpy(R, a, X, b, size);
	
	std::parallel_launch({size}, saxpy_kernel, D, a, X, b, size);
	
	for(int i = 0; i < size; ++i)
	{
		std::cout << "R[" << i <<"] = " << R[i] << "\n";
		std::cout << "D[" << i <<"] = " << D[i] << "\n";
	}
	
	return 0;
}

