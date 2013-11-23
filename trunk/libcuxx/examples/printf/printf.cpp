
#include <__parallel_config>
#include <cstdio>

int main(int argc, char** argv)
{
	std::printf("Hello GPU\n");

	for(int i = 0; i < argc; ++i)
	{
		std::printf("Argument[%d] = '%s'\n", i, argv[i]);
	}

	return 0;
}



