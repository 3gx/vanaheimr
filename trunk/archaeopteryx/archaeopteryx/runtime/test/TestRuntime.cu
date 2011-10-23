/*! \file TestRuntime.cu
 *  \date   Tuesday September 27, 2011
 *  \author Sudnya Padalikar
 *  <mailsudnya@gmail.com>
 *  \brief  A test file to test the implementation of Runtime.
 *   */

#include <archaeopteryx/runtime/interface/Runtime.h>

// High level:
// Create i/p data, load binary, launch threads/run kernel, read outputs and verify outputs 
// Detailed steps:
// allocate memory for i/p data 
// maybe add memset/memcpy in runtime?
// Runtime loads binary from a given filename 
// Runtime finds the entry pt function
// simulation is launched by kernel for that PC
// runtime should memcpy outputs
// test should have a reference function to compare the output against

#define ARRAY_SIZE 1024

extern "C" __device__ void runTest()
{
    unsigned int* refX = 0;
    unsigned int* refY = 0;
    unsigned int a     = 5;

    refX = (unsigned int*)malloc(ARRAY_SIZE);
    refY = (unsigned int*)malloc(ARRAY_SIZE);

    util::HostReflection::launch(1, ARRAY_SIZE, __FILE__, "initValues",
    	util::HostReflection::createPayload(refX));
    util::HostReflection::launch(1, ARRAY_SIZE, __FILE__, "initValues",
    	util::HostReflection::createPayload(refY));
    util::HostReflection::launch(1, ARRAY_SIZE, __FILE__, "refCudaSaxpy",
    	util::HostReflection::createPayload(refX, refY, a));

    //allocate memory for arrays used by saxpy
    size_t baseX = 0;
    size_t baseY = baseX + ARRAY_SIZE;
    bool allocX  = rt::Runtime::allocateMemoryChunk(ARRAY_SIZE, baseX);
    bool allocY  = rt::Runtime::allocateMemoryChunk(ARRAY_SIZE, baseY);

    if (allocX && allocY)
    {
        util::HostReflection::launch(1, ARRAY_SIZE, __FILE__, "initValues", 
        	util::HostReflection::createPayload(
        	rt::Runtime::translateCudaAddressToSimulatedAddress((void*)baseX)));
        util::HostReflection::launch(1, ARRAY_SIZE, __FILE__, "initValues", 
        	util::HostReflection::createPayload(
        	rt::Runtime::translateCudaAddressToSimulatedAddress((void*)baseY)));
		rt::Runtime::loadBinary("saxpy.cu");
        rt::Runtime::setupKernelEntryPoint("main");
        rt::Runtime::launchSimulation();
        void* translatedAddress =
        	rt::Runtime::translateSimulatedAddressToCudaAddress((void*)baseY);
        util::HostReflection::launch(1, 1, __FILE__, "compareMemory",
        	util::HostReflection::createPayload(translatedAddress,
        	refY, ARRAY_SIZE));
    }

} 

extern "C" __device__ void compareMemory(util::HostReflection::Payload& payload)
{
	unsigned int* result       = payload.get<unsigned int*>(0);
    unsigned int* ref          = payload.get<unsigned int*>(1);
    unsigned int  memBlockSize = payload.get<unsigned int >(2);
    
    for (unsigned int i = 0; i < memBlockSize; ++i)
    {
        if (ref[i] != result[i])
        {
            printf("Memory not equal\n");
            return;
        }
    }
}

extern "C" __device__ void initValues(util::HostReflection::Payload& payload)
{
	unsigned int* array = payload.get<unsigned int*>(0);

    array[threadIdx.x] = threadIdx.x;
}

__device__ void refCudaSaxpy(util::HostReflection::Payload& payload)
{
	unsigned int* y = payload.get<unsigned int*>(0);
	unsigned int* x = payload.get<unsigned int*>(1);
	unsigned int  a = payload.get<unsigned int >(2);
	
    y[threadIdx.x] = a*x[threadIdx.x] + y[threadIdx.x];
}

__global__ void launchTest()
{
    util::HostReflection::launch(1, 1, __FILE__, "runTest");
}

int main(int argc, char** argv)
{
    util::HostReflection::create();

	launchTest<<<1, 1>>>();
	
    util::HostReflection::destroy();
}


