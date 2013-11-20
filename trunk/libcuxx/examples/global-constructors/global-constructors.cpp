
#include <__parallel_config>
#include <cstdio>

class Initializer
{
public:
	Initializer()
		{ std::printf("Running global constructor for Initializer\n"); }

};

Initializer initializer;

int main(int argc, char** argv)
{
	return 0;
}




