
/*! \file   CudaUtilities.h
	\date   Saturday Feburary 26, 2011
	\author Gregory Diamos and Sudnya Diamos
		<gregory.diamos@gatech.edu, mailsudnya@gmail.com>
	\brief  A set of common CUDA functions.
*/

#pragma once

// Standard Library Includes
#include <cstring>

/*! \brief Common utility functions */
namespace util
{

template<typename T>
__device__ T getParameter(void* parameter, unsigned int byte = 0)
{
	return *(T*)((char*)parameter + byte);
}

const unsigned int hostBufferSize = 4096;

typedef void (*AsyncFunctionPointer)(void*);

class AsyncFunctionRecord
{
public:
	const char* name;
	AsyncFunctionPointer function;
};

__device__ void* hostBuffer;
__device__ AsyncFunctionRecord functionTable[] = {
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 },
		{ 0, 0 }
		};

__device__ unsigned int strlen(const char* s)
{
	const char* i = s;
	while(++i);
	
	return i - s;
}

__device__ inline void async_system_call(const char* name,
	void* p1 = 0, void* p2 = 0)
{
	unsigned int* offset = (unsigned int*)hostBuffer;
	
	unsigned int packetSizeOffset = *offset;
	unsigned int startingOffset   = packetSizeOffset + sizeof(unsigned int);
	char* dataBase = (char*)(offset + 1);

	unsigned int size = strlen(name) + 1;

	std::memcpy(dataBase + startingOffset, name, size);
	unsigned int currentOffset = startingOffset + size;
	
	if(p1 != 0)
	{
		std::memcpy(dataBase + currentOffset, &p1, sizeof(void*));
		currentOffset += sizeof(void*);
	}

	if(p2 != 0)
	{
		std::memcpy(dataBase + currentOffset, &p2, sizeof(void*));
		currentOffset += sizeof(void*);
	}
	
	*offset = currentOffset;
	
	size = currentOffset - startingOffset;
	std::memcpy(dataBase + packetSizeOffset, &size, sizeof(unsigned int));
}

inline void setupHostReflection()
{
	unsigned int* buffer = 0;
	cudaHostAlloc(&buffer, hostBufferSize, cudaHostAllocDefault);
	
	*buffer = 0;
	
	cudaMemcpyToSymbol("hostBuffer", &buffer, sizeof(void*));
}

__device__ int strcmp(const char* str1, const char* str2)
{
	while(*str1 && *str2)
	{
		if(*str1 != *str2) return 1;
		
		++str1;
		++str2;
	}
	
	if(!*str1 && *str2) return 0;
	
	return 1;
}

__global__ void dispatch(const char* name, void* payload)
{
	for(AsyncFunctionRecord* record = functionTable;
		record->name != 0; ++record)
	{
		if(strcmp(record->name, name) == 0)
		{
			record->function(payload);
		}
	}
}

inline void teardownHostReflection()
{
	void* hostBuffer = 0;
	cudaMemcpyFromSymbol(&hostBuffer, "hostBuffer", sizeof(void*));

	unsigned int* hostBufferBase = (unsigned int*)hostBuffer;
	unsigned int totalSize  = *hostBufferBase;
	unsigned int packetSize = 0;
	
	char* dataBase = (char*)(hostBufferBase + 1);
	
	for(unsigned int i = 0; i < totalSize; i += packetSize)
	{
		packetSize = *(unsigned int*)(dataBase + i);
		const char* name = dataBase + i + sizeof(unsigned int);
		
		const char* payload = name + std::strlen(name) + 1;
		
		dispatch<<<1, 1, 1>>>(name, (void*)payload);
	}
	
	cudaFreeHost(hostBuffer);
}

}

