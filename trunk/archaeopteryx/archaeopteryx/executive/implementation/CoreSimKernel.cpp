/*! \file   CoreSimKernel.h
 *  \date   Thursday September 15, 2011
 *  \author Sudnya Padalikar
 *  <mailsudnya@gmail.com>
 *  \brief  The implementation file for the Core simulator of the Kernel class.
 *   */

#include <archaeopteryx/executive/interface/CoreSimKernel.h>
#include <archaeopteryx/executive/interface/CoreSimBlock.h>
#include <archaeopteryx/ir/interface/Binary.h>

namespace executive
{

__device__ void CoreSimKernel::launchKernel(unsigned int simulatedBlocks, CoreSimBlock* blocks, ir::Binary* binary)
{
    for (unsigned int simulatedBlock = blockIdx.x;
    	simulatedBlock < simulatedBlocks; simulatedBlock += gridDim.x)
    {
        // TODO set block state
        blocks[blockIdx.x].setupBinary(binary);
        blocks[blockIdx.x].setupCoreSimBlock();
        blocks[blockIdx.x].runBlock();
    }
    
}

}
