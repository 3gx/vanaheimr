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

__global__ void runTest()
{
    unsigned int* refX;
    unsigned int* refY;
    unsigned int a = 5;
    refX = malloc(ARRAY_SIZE);
    refY = malloc(ARRAY_SIZE);
    util::HostReflection::launch(1, ARRAY_SIZE, "initValues",refX);
    util::HostReflection::launch(1, ARRAY_SIZE, "initValues",refY);
    util::HostReflection::launch(1, ARRAY_SIZE, "refCudaSaxpy", refX, refY, a);

    //allocate memory for arrays used by saxpy
    unsigned int baseX = 0;
    unsigned int baseY = baseX + ARRAY_SIZE;
    bool allocX        = Runtime::allocateMemoryChunk(ARRAY_SIZE, baseX);
    bool allocY        = Runtime::allocateMemoryChunk(ARRAY_SIZE, baseY);
    if (allocX && allocY)
    {
        util::HostReflection::launch(1, ARRAY_SIZE, "initValues", Runtime::translateCudaAddressToSimulatedAddress(baseX));
        util::HostReflection::launch(1, ARRAY_SIZE, "initValues", Runtime::translateCudaAddressToSimulatedAddress(baseY));
        Runtime::loadBinary("saxpy.cu");
        Runtime::setupKernelEntryPoint("main");
        Runtime::launchSimulation();
        void* translatedAddress = Runtime::translateSimulatedAddressToCudaAddress(baseY);
        util::HostReflection::launch(1, 1, "compareMemory", translatedAddress, refY, ARRAY_SIZE);
    }

} 

__device__ void compareMemory(void* ref, void* result, unsigned int memBlockSize)
{
    for (unsigned int i = 0; i < memBlockSize; ++i)
    {
        if (ref[i] != result[i])
        {
            printf("Memory not equal\n");
            return;
        }
    }
}

__device__ void initValues(unsigned int* array)
{
    array[threadIdx.x] = threadIdx.x;
}

__device__ void refCudaSaxpy(unsigned int* y, unsigned int* x, unsigned int a)
{
    y[threadIdx.x] = a*x[threadIdx.x] + y[threadIdx.x];
}

int main(int argc, char** argv)
{
    util::HostReflection::create();
    runTest<<<1, 1, 0>>>();

    util::HostReflection::destroy();

}
