/*! \file   CoreSimBlock.cpp
	\date   Sunday August, 7th 2011
	\author Sudnya Diamos
		<mailsudnya@gmail.com>
	\brief  The implementation file for the Core simulator of the thread block class.
*/

#include <archaeopteryx/executive/interface/CoreSimBlock.h>

namespace executive
{

__device__ CoreSimBlock(BlockState* blockState, ir::Binary* binary)
: m_blockState(blockState), m_binary(binary)
{
    m_registerFiles = new Register[m_blockState->registersPerThread * m_blockState->threadsPerBlock];
    m_sharedMemory = new SharedMemory[m_blockState->sharedMemoryPerBlock];
    m_localMemory = new LocalMemory[m_blockState->localMemoryPerThread];

    m_threadIdInWarp = threadIdx.x % WARP_SIZE;
    m_threads = new CoreSimThreads[m_blockState->threadsPerBlock];
}

__device__ bool areAllThreadsFinished()
{
    //TODO evaluate some bool 'returned' for all threads
    bool finished = m_warp[m_threadIdInWarp].finished;
    __shared__ bool tempFinished[WARP_SIZE];

    tempFinished[m_threadIdInWarp] = finished;
    // barrier

    for(unsigned int i = 2; i < WARP_SIZE; i*=2)
    {
        if(m_threadIdInWarp % i == 0)
        {
            finished &= tempFinished[m_threadIdInWarp + i/2];
        }
        // barrier
        
        if(m_threadIdInWarp % i == 0)
        {
            tempFinished[m_threadIdInWarp] = finished;
        }

        // barrier
    }
    
    finished = tempFinished[0];

    return finished;
}

__device__ void roundRobinScheduler()
{
    if (m_threadIdInWarp == 0)
    {
        if (m_warp - m_threads + WARP_SIZE > m_blockState->threadsPerBlock)
        {
            m_warp = m_threads;
        }
        else
        {
            m_warp += WARP_SIZE;
        }
    }
    //barrier
}

__device__ findNextPC()
{
    //TO DO - decide how to handle branches?
}

__device__ bool setPredicateMaskForWarp(PC pc)
{
    //TO DO - evaluate a predicate over the entire warp
    return pc == m_warp[m_threadIdInWarp].pc;
}

__device__ fetchInstruction()
{

}

__device__ executeWarp(InstructionContainer* instruction, PC pc)
{
    bool predicateMask = setPredicateMaskForWarp(pc);    
    
    //some function for all threads if predicateMask is true
    if(predicateMask)
    {
        m_warp[m_threadIdInWarp]->executeInstruction(instruction, pc);
    }
}

// Entry point to the block simulation
// It performs the following operations
//   1) Schedule group of simulated threads onto CUDA warps (static/round-robin)
//   2) Pick the next PC to execute (the one with the highest priority using a reduction)
//   3) Set the predicate mask (true if threadPC == next PC, else false)
//   4) Fetch the instruction at the selected PC
//   5) Execute all threads with true predicate masks
//   6) Save the new PC, goto 1 if all threads are not done
 __device__ void runBlock()
{
    while (!areAllThreadsFinished())
    {
        roundRobinScheduler();
        PC nextPC = findNextPC();
        InstructionContainer* instruction = fetchInstruction(nextPC);
        executeWarp(instruction, nextPC);
    }
}

}
