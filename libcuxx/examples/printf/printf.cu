
#include <cstdio>

__device__ int _main(int argc, char** argv)
{
	std::printf("Hello GPU\n");
	std::printf("Hello GPU %d %d", 1, 2);
	
	return 0;
}


