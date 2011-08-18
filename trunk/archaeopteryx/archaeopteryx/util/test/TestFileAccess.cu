/*! \file   TestFileAccesses.h
	\date   Tuesday June 28, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the TestFileAccesses series of units tests for
		CUDA file accesses.
*/

// Archaeopteryx Includes
#include <archaeopteryx/util/interface/File.h>
#include <archaeopteryx/util/interface/HostReflection.h>

// Standard Library Includes
#include <string>
#include <iostream>

namespace test
{

__global__ void kernelTestReadWriteFile(const char* filename,
	void* result, const void* data, unsigned int size)
{
	util::File file(filename);
	
	file.write(data, size);
	file.read(result, size);
}

bool testReadWriteFile(const std::string& filename, unsigned int size)
{
	char* deviceFilename = 0;
	cudaMallocHost(&deviceFilename, filename.size() + 1);

	unsigned int* deviceData = 0;
	cudaMallocHost(&deviceData, size);

	unsigned int* deviceResult = 0;
	cudaMallocHost(&deviceResult, size);

	for(unsigned int i = 0; i < size; ++i)
	{
		deviceData[i] = std::rand();
	}

	kernelTestReadWriteFile<<<1, 1>>>(deviceFilename, deviceResult,
		deviceData, size);
	
	bool pass = std::memcmp(deviceData, deviceResult, size) == 0;
	
	cudaFreeHost(deviceFilename);
	cudaFreeHost(deviceData);
	cudaFreeHost(deviceResult);
	
	return pass;
}

}

int main(int argc, char** argv)
{
	util::HostReflection::create();

	if(test::testReadWriteFile("Archaeopteryx_Test_File", 1 << 20))
	{
		std::cout << "Pass/Fail: Pass\n";
	}
	else
	{
		std::cout << "Pass/Fail: Fail\n";
	}

	util::HostReflection::destroy();
}


