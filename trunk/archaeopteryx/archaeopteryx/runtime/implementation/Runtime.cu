/*! \file   Runtime.cpp
 *   \date   Tuesday Sept, 13th 2011
 *   \author Sudnya Padalikar
 *   <mailsudnya@gmail.com>
 *   \brief  The implementation file for the runtime API.
 **/

// Archaeopteryx Includes
#include <archaeopteryx/executive/interface/CoreSimKernel.h>
#include <archaeopteryx/executive/interface/CoreSimBlock.h>

#include <archaeopteryx/runtime/interface/Runtime.h>
#include <archaeopteryx/runtime/interface/MemoryPool.h>

namespace archaeopteryx
{

namespace rt
{

class RuntimeState
{
public:
	typedef util::vector<executive::CoreSimBlock> CTAVector;
	typedef util::map<util::string, ir::Binary>   BinaryMap;
	typedef executive::CoreSimKernel              Kernel;

public:
	Kernel     kernel;
	CTAVector  hardwareCTAs;
	BinaryMap  binaries;
	MemoryPool memory;
	
public:
	size_t parameterMemoryAddress;
	
public:
	size_t simulatedBlockCount;
	size_t programEntryPointAddress;

};

__device__ static RuntimeState* state = 0;

__device__ void Runtime::create()
{
	state = new RuntimeState;

	state->parameterMemoryAddress = allocateMemoryWhereAvailable(
		util::KnobDatabase::getKnob<size_t>("parameter-memory-size"),
		parameterMemory);

	unsigned int ctas = util::KnobDatabase::getKnob<unsigned int>("simulator-ctas");
	state->hardwareCTAs.resize(ctas);
}

__device__ void Runtime::destroy()
{
	delete state; state = 0;
}

__device__ void Runtime::loadBinary(const char* fileName)
{
    state->binaries.insert(util::make_pair(fileName, ir::Binary(fileName)));
}

__device__ bool Runtime::mmap(size_t bytes, Address address)
{
	return state->memory.allocate(bytes, address);
}

__device__ Runtime::Address Runtime::mmap(size_t bytes)
{
	return state->memory.allocate(bytes);
}

__device__ void Runtime::munmap(size_t address)
{
	state->memory->deallocate(address);
}

__device__ void* Runtime::translateCudaAddressToSimulatedAddress(void* cudaAddress)
{
	return state->memory.translateAddress((size_t) simAddress);
}

// The Runtime class owns all of the simulator state, it should have allocated it in the constructor
//  a) simulated state is CoreSimKernel/Block/Thread and other classes
//  b) this call changes the number of CoreSimBlock/Thread
__device__ void Runtime::setupLaunchConfig(unsigned int totalCtas, unsigned int threadsPerCta)
{
    state->simulatedBlockCount = totalCtas;
   
	// TODO: run in a kernel 
    for(RuntimeState::CTAVector::iterator cta = state->hardwareCTAs.begin();
		cta != state->hardwareCTAs.end(); ++cta)
    {
        cta->setNumberOfThreadsPerBlock(threadsPerCta);
    }
}

// Similar to the previous call, this sets the memory sizes
__device__ void Runtime::setupMemoryConfig(unsigned int localMemoryPerThread, unsigned int sharedMemoryPerCta)
{
	// TODO: run in a kernel 
    for(RuntimeState::CTAVector::iterator cta = state->hardwareCTAs.begin();
		cta != state->hardwareCTAs.end(); ++cta)
    {
        cta->setMemoryState(localMemoryPerThread, sharedMemoryPerCta);
    }
}

__device__ void Runtime::setupArgument(const void* data, size_t size, size_t offset)
{
	char* parameterBase =
		(char*)translateCudaAddressToSimulatedAddress(state->parameterMemoryAddress);
	
	std::memcpy(parameterBase + offset, data, size);
}

// Set the PC of all threads to the PC of the specified function
//   Call into the binary to get the PC
__device__ void Runtime::setupKernelEntryPoint(const char* functionName)
{
    state->programEntryPointAddress = findFunctionsPC(functionName);    
}

// Start a new asynchronous kernel with the right number of HW CTAs/threads
__device__ void Runtime::launchSimulation()
{
	unsigned int ctas    = util::KnobDatabase::getKnob<unsigned int>("simulator-ctas");
	unsigned int threads = util::KnobDatabase::getKnob<unsigned int>("simulator-threads-per-cta");
	
	launchSimulationInParallel<<<ctas, threads>>>();
}

__global__ void Runtime::launchSimulationInParallel()
{
    state->kernel.launchKernel(state->simulatedBlockCount, 	
        state->hardwareCTAs, getSelectedBinary());
}

__device__ void Runtime::unloadBinaries()
{
	state->binaries.clear();
}

__device__ size_t Runtime::findFunctionsPC(const char* functionName)
{
	for(State::BinaryMap::iterator binary = state->binaries.begin();
		binary != state->binaries.end(); ++binary)
	{
		if(!binary->second.containsFunctin(functionName)) continue;

		return binary->second.findFunctionsPC(functionName);
	}

	assertM(false, "Function name not found.");

	return 0;
}

__device__ ir::Binary* Runtime::getSelectedBinary()
{
	//TODO support multiple binaries (requires linking)
	return &state->binaries->begin()->second;
}

}

}

