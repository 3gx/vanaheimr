


void saxpy(int* R, int* A, int* X, int b, int size)
{
	for(int i = 0; i < size; ++i)
	{
		A[i] = A[i] * X[i] + b;
	}
}

void saxpy_kernel()

extern void launch_kernel();

int main(int argc, char** argv)
{
	int A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
	int X[] = {1, 2, 3, 5, 6, 7, 9, 4, 8};
	int b = 2;
	
	int size = sizeof(A);
	
	int R[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	int D[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	
	saxpy(R, A, X, b, size);
	
	launch_cuda_kernel();
	
/*	for(int i = 0; i < size; ++i)
	{
		std::cout << "R[" << i <<"] = " << R[i] << "\n";
		std::cout << "D[" << i <<"] = " << R[i] << "\n";
	}
*/	
	return 0;
}



