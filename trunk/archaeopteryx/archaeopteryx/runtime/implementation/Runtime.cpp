/*! \file   Runtime.cpp
 *   \date   Tuesday Sept, 13th 2011
 *   \author Sudnya Padalikar
 *   <mailsudnya@gmail.com>
 *   \brief  The implementation file for the runtime API.
 **/

#include <archaeopteryx/executive/interface/CoreSimKernel.h>
#include <archaeopteryx/runtime/interface/Runtime.h>

#define NUMBER_OD_HW_THREADS_PER_BLOCK 128
#define NUMBER_OF_HW_BLOCKS 64
#define PHYSICAL_MEMORY_SIZE (1 << 20)

namespace rt
{

__device__ Runtime::Runtime()
{
    m_blocks         = new executive::CoreSimBlock[NUMBER_OF_HW_BLOCKS];i
    m_physicalMemory = new char[PHYSICAL_MEMORY_SIZE];
}

__device__ Runtime::~Runtime()
{
   delete m_blocks[];
}

// We will need a list/map of open binaries
//  a) maybe just one to start with
//  b) create a binary object using the filename in the constructor, it
//     will read the data for us
__device__ void Runtime::loadBinary(const char* fileName)
{
    m_loadedBinary = new Binary(fileName);
    //TODO: eventually m_loadedBinary.push_back(new Binary(fileName));
}


// We want a window of allocated global memory (malloced)
//   a) This window contains all allocations
//   b) The base of the window starts at the address of the first allocation
//   c) All other allocations are offsets from the base
//
//   It looks like this
//   
//   base 
//   <------------------------------------------------------------>
//   <------>                      <------------------->
//   allocation1 (address=base)    allocation2 (address=base+offset)
//
__device__ bool Runtime::allocateMemory(size_t bytes, size_t address)
{
    return !(address+bytes > PHYSICAL_MEMORY_SIZE);
}

// The Runtime class owns all of the simulator state, it should have allocated it in the constructor
//  a) simulated state is CoreSimKernel/Block/Thread and other classes
//  b) this call changes the number of CoreSimBlock/Thread
__device__ void Runtime::setupLaunchConfig(unsigned int totalCtas, unsigned int threadsPerCta)
{
    m_simulatedBlocks = totalCtas;
    
    for (unsigned int i = 0; i < NUMBER_OF_HW_BLOCKS; ++i)
    {
        m_blocks[i]->setNumberOfThreadsPerBlock(threadsPerCta);
    }
}

// Similar to the previous call, this sets the memory sizes
__device__ void Runtime::setupMemoryConfig(unsigned int localMemoryPerThread, unsigned int sharedMemoryPerCta)
{
    for (unsigned int i = 0; i < NUMBER_OF_HW_BLOCKS; ++i) 
    {
        m_blocks[i]->setMemoryState(localMemoryPerThread, sharedMemoryPerCta);
    }
}

// Set the PC of all threads to the PC of the specified function
//   Call into the binary to get the PC
__device__ void Runtime::setupKernelEntryPoint(const char* functionName)
{
    m_launchSimulationAtPC = m_loadedBinary->findFunctionsPC(functionName);    
}

// Start a new asynchronous kernel with the right number of HW CTAs/threads
__device__ void Runtime::launchSimulation()
{
    util::HostReflection::launch(NUMBER_OF_HW_BLOCKS, NUMBER_OF_HW_THREADS_PER_BLOCK, "Runtime::launchSimulationInParallel" );
}

__device__ void Runtime::launchSimulationInParallel()
{
    m_kernel->launchKernel(m_simulatedBlocks, m_blocks);
}


__device__ void munmap(size_t address)
{
}

__device__ void unloadBinary(const char* fileName)
{
    delete m_loadedBinary;
}


}

