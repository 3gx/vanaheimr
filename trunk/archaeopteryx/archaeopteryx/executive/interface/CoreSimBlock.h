/*! \file   CoreSimBlock.h
	\date   Saturday Feburary 23, 2011
	\author Gregory and Sudnya Diamos
		<gregory.diamos@gatech.edu, mailsudnya@gmail.com>
	\brief  The header file for the Core simulator of the thread block class.
*/

#pragma once
#define WARP_SIZE     32
#define LOG_WARP_SIZE 5

//Forward declarations
namespace executive { class BlockState; }
namespace ir        { class Binary; }
/*! \brief A namespace for program execution */
namespace executive
{
class CoreSimBlock
{

    private:
        //FetchUnit m_fetchUnit;
        typedef unsigned long long Register;
        Register* m_registerFiles;
        BlockState* m_blockState;
        SharedMemory* m_sharedMemory;
        LocalMemory* m_localMemory;
        CoreSimThread* m_threads;
        typedef CoreSimThread* Warp;
	Warp m_warp;
        bool m_predicateMask[WARP_SIZE]; 
	unsigned int m_threadIdInWarp;

    public:
        //public members
        class BlockState
        {
            public:
                unsigned int blockId;
                unsigned int registersPerThread;
                unsigned int localMemoryPerThread;
                unsigned int threadsPerBlock;
                unsigned int sharedMemoryPerBlock;
        };
    public:
    	// Initializes the state of the block
    	//  1) Register file
    	//  2) shared memory 
    	//  3) local memory for each thread
    	//  4) thread contexts
        __device__ CoreSimBlock(BlockState* blockState, ir::Binary* binary);
    
    public:
    	// Entry point to the block simulation
    	//  It performs the following operations
    	//   1) Schedule group of simulated threads onto CUDA warps (static/round-robin)
    	//   2) Pick the next PC to execute (the one with the highest priority using a reduction)
    	//   3) Set the predicate mask (true if threadPC == next PC, else false)
    	//   4) Fetch the instruction at the selected PC
    	//   5) Execute all threads with true predicate masks
    	//   6) Save the new PC, goto 1 if all threads are not done
    	__device__ void runBlock();
    
    public:
    	// Interfaces to CoreSimThread
        __device__ CoreSimThread* getCoreSimThread(unsigned int id);
        __device__ unsigned int getSimulatedThreadCount();
        __device__ CoreSimThread::Value getRegister(unsigned int, unsigned int);
        __device__ void setRegister(unsigned int, unsigned int, const CoreSimThread::Value&);
        __device__ CoreSimThread::Value translateVirtualToPhysical(const CoreSimThread::Value);
        __device__ void barrier(unsigned int);
        __device__ unsigned int returned(unsigned int, unsigned int);
};

}

