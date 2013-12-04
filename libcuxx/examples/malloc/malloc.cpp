
#include <__parallel_config>
#include <cstdlib>
#include <cassert>

int main(int argc, char** argv)
{
	int* allocation = reinterpret_cast<int*>(std::malloc(sizeof(int) * argc));

	for(int i = 0; i < argc; ++i)
	{
		allocation[i] = i;
	}
	
	int result = 0;

	for(int i = 0; i < argc; ++i)
	{
		result += allocation[i];
	}

	std::free(allocation);

	return result;
}


