/*! \file   Runtime.cpp
 *   \date   Tuesday Sept, 13th 2011
 *   \author Sudnya Padalikar
 *   <mailsudnya@gmail.com>
 *   \brief  The implementation file for the runtime API.
 **/

#include <archaeopteryx/executive/interface/CoreSimKernel.h>
#include <archaeopteryx/runtime/interface/Runtime.h>

namespace rt
{

__device__ void Runtime::loadBinary(const char* fileName)
{
}

__device__ void* Runtime::allocateMemory(size_t bytes, size_t address)
{
    return ((void*)malloc(bytes));
}

__device__ void Runtime::setupLaunchConfig(unsigned int totalCtas, unsigned int threadsPerCta)
{
}

__device__ void Runtime::setupMemoryConfig(unsigned int localMemoryPerThread, unsigned int sharedMemoryPerCta)
{
}

__device__ void Runtime::setupKernelEntryPoint(size_t startAddress)
{
}

__device__ void Runtime::launchSimulation()
{
}

__device__ void munmap(size_t address)
{
}

__device__ void unloadBinary(const char* fileName)
{
}


}

