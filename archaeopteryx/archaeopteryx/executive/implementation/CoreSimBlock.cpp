/*! \file   CoreSimBlock.cpp
	\date   Sunday August, 7th 2011
	\author Sudnya Padalikar
		<mailsudnya@gmail.com>
	\brief  The implementation file for the Core simulator of the thread block class.
*/

#include <archaeopteryx/executive/interface/CoreSimBlock.h>
#include <archaeopteryx/ir/interface/Instruction.h>

namespace executive
{

__device__ CoreSimBlock::CoreSimBlock(BlockState* blockState, ir::Binary* binary)
: m_blockState(blockState), m_binary(binary)
{
    m_registerFiles  = new Register[m_blockState->registersPerThread * m_blockState->threadsPerBlock];
    m_sharedMemory   = new SharedMemory[m_blockState->sharedMemoryPerBlock];
    m_localMemory    = new LocalMemory[m_blockState->localMemoryPerThread];

    m_threadIdInWarp = threadIdx.x % WARP_SIZE;
    m_threads        = new CoreSimThread[m_blockState->threadsPerBlock];
}

__device__ bool CoreSimBlock::areAllThreadsFinished()
{
    //TODO evaluate some bool 'returned' for all threads
    bool finished = m_warp[m_threadIdInWarp].finished;
    __shared__ bool tempFinished[WARP_SIZE];

    tempFinished[m_threadIdInWarp] = finished;
    // barrier

    for (unsigned int i = 2; i < WARP_SIZE; i*=2)
    {
        if (m_threadIdInWarp % i == 0)
        {
            finished = finished & tempFinished[m_threadIdInWarp + i/2];
        }
        // barrier
        
        if (m_threadIdInWarp % i == 0)
        {
            tempFinished[m_threadIdInWarp] = finished;
        }

        // barrier
    }
    
    finished = tempFinished[0];

    return finished;
}

__device__ void CoreSimBlock::roundRobinScheduler()
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

__device__ unsigned int CoreSimBlock::findNextPC(unsigned int& returnPriority)
{
    __shared__ uint2 priority[WARP_SIZE];
    unsigned int localThreadPriority = 0;
    unsigned int localThreadPC       = 0;

    // only give threads a non-zero priority if they are NOT waiting at a barrier
    if (m_warp[m_threadIdInWarp].barrierBit == false)
    {
        localThreadPriority = m_warp[m_threadIdInWarp].instructionPriority;
        localThreadPC       = m_warp[m_threadIdInWarp].pc;

        priority[m_threadIdInWarp].x = localThreadPriority;
        priority[m_threadIdInWarp].y = localThreadPC;
    }
    
    // warp_barrier

    for (unsigned int i = 2; i < WARP_SIZE; i*=2)
    {
        if (m_threadIdInWarp % i == 0)
        {
            unsigned int neighborsPriority = priority[m_threadIdInWarp + i/2].x;
            unsigned int neighborsPC       = priority[m_threadIdInWarp + i/2].y;

            bool local = localThreadPriority > neighborsPriority;

            localThreadPriority = local ? localThreadPriority : neighborsPriority;
            localThreadPC       = local ? localThreadPC       : neighborsPC;
        }
        // warp_barrier
        if (m_threadIdInWarp % i == 0)
        {
            priority[m_threadIdInWarp].x = localThreadPriority;
            priority[m_threadIdInWarp].y = localThreadPC;
        }
        // warp_barrier
    }

    unsigned int maxPriority = priority[0].x;
    unsigned int maxPC       = priority[0].y;
 
    returnPriority = maxPriority;

    return maxPC;
}

__device__ bool CoreSimBlock::setPredicateMaskForWarp(PC pc)
{
    //TO DO - evaluate a predicate over the entire warp
    return pc == m_warp[m_threadIdInWarp].pc;
}

__device__ CoreSimBlock::InstructionContainer CoreSimBlock::fetchInstruction(PC pc)
{
    __shared__ InstructionContainer instruction;
    
    if (m_threadIdInWarp == 0)
    {
        m_binary->copyCode(&instruction, pc, 1);
    }
    // barrier
    return instruction;
}

__device__ void CoreSimBlock::executeWarp(InstructionContainer* instruction, PC pc)
{
    bool predicateMask = setPredicateMaskForWarp(pc);    
    
    //some function for all threads if predicateMask is true
    if (predicateMask)
    {
        m_warp[m_threadIdInWarp].pc = m_warp[m_threadIdInWarp].executeInstruction(&instruction->asInstruction, pc);
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
 __device__ void CoreSimBlock::runBlock()
{
    unsigned int executedCount = 0;
    unsigned int scheduledCount = 0;

    while (!areAllThreadsFinished())
    {
        roundRobinScheduler();
        unsigned int priority = 0;
        ++scheduledCount;
        PC nextPC = findNextPC(priority);
        // only execute if all threads in this warp are NOT waiting on a barrier
        if (priority != 0)
        {
             InstructionContainer instruction = fetchInstruction(nextPC);
             executeWarp(&instruction, nextPC);
             ++executedCount;
        }

        if (scheduledCount == m_blockState->threadsPerBlock / WARP_SIZE)
        {
            if (executedCount == 0)
            {
                clearAllBarrierBits();
            }
            scheduledCount = 0;
            executedCount  = 0;
        }
    }
}

__device__ CoreSimThread::Value CoreSimBlock::getRegister(unsigned int threadId, unsigned int reg)
{
    return m_registerFiles[(m_blockState->registersPerThread * threadId)+reg];
}

__device__ void CoreSimBlock::setRegister(unsigned int reg, unsigned int threadId, const CoreSimThread::Value& result)
{
    m_registerFiles[(m_blockState->registersPerThread*threadId)+reg] = result;
}

__device__ CoreSimThread::Value CoreSimBlock::translateVirtualToPhysical(const CoreSimThread::Value v)
{
    return v; // we will modify this to something much sophisticated later
}


__device__ void CoreSimBlock::barrier(unsigned int threadId)
{
    m_threads[threadId].barrierBit = true;
}

__device__ unsigned int CoreSimBlock::returned(unsigned int threadId, unsigned int pc)
{
    m_threads[threadId].finished = true;

    // TODO return the PC from the stack
    return 0;
}

__device__ void CoreSimBlock::clearAllBarrierBits()
{
    for (unsigned int i = 0 ; i < (m_blockState->threadsPerBlock)/WARP_SIZE ; ++i)
    {
        unsigned int logicalThread = i * WARP_SIZE + m_threadIdInWarp;
	m_threads[logicalThread].barrierBit = false;
        //barrier should be here but it is slow (every warp)
    } 
    //barrier -> we gurantee that we wont clobber values (blocks are not overlapping)
}

}
