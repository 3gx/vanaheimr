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
}

}
