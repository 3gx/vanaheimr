/*! \file   CoreSimKernel.h
 *  \date   Thursday September 15, 2011
 *  \author Sudnya Padalikar
 *  <mailsudnya@gmail.com>
 *  \brief  The implementation file for the Core simulator of the Kernel class.
 *   */

#include <archaeopteryx/executive/interface/CoreSimKernel.h>

namespace executive
{

__device__ void CoreSimKernel::launchKernel(unsigned int simulatedBlocks, CoreSimBlock* blocks)
{
    for (unsigned int simulatedBlock = blockIdx.x; simulatedBlock < m_simulatedBlocks; simulatedBlock += gridDim.x)
    {
        blocks[blockIdx.x]->setBlockId(simulatedBlock);
        blocks[blockIdx.x]->runBlock();
    }
    
}

}
