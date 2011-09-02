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
	char* hostFilename = 0;
	char* deviceFilename = 0;
	cudaHostAlloc(&hostFilename, filename.size() + 1, cudaHostAllocMapped);
	cudaHostGetDevicePointer(&deviceFilename, hostFilename, 0);

	strcpy(hostFilename, filename.c_str());

	unsigned int* hostData = 0;
	unsigned int* deviceData = 0;
	cudaHostAlloc(&hostData, size, cudaHostAllocMapped);
	cudaHostGetDevicePointer(&deviceData, hostData, 0);

	unsigned int* hostResult = 0;
	unsigned int* deviceResult = 0;
	cudaHostAlloc(&hostResult, size, cudaHostAllocMapped);
	cudaHostGetDevicePointer(&deviceData, hostData, 0);

	for(unsigned int i = 0; i < size/sizeof(unsigned int); ++i)
	{
		hostData[i] = std::rand();
	}
	
	kernelTestReadWriteFile<<<1, 1>>>(deviceFilename, deviceResult,
		deviceData, size);
	
	bool pass = std::memcmp(hostData, hostResult, size) == 0;
	
	cudaFreeHost(hostFilename);
	cudaFreeHost(hostData);
	cudaFreeHost(hostResult);
	
	return pass;
}

}

int main(int argc, char** argv)
{
	util::HostReflection::create();

	if(test::testReadWriteFile("Archaeopteryx_Test_File", 10))
	{
		std::cout << "Pass/Fail: Pass\n";
	}
	else
	{
		std::cout << "Pass/Fail: Fail\n";
	}

	util::HostReflection::destroy();
}


