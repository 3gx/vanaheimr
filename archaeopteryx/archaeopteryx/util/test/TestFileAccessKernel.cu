

#include <archaeopteryx/util/interface/File.h>

__global__ void deviceMain(const char* filename, void* result,
	const void* data, unsigned int size)
{
	archaeopteryx::util::File file(filename);

	file.write(data, size);
	file.read(result, size);
}

