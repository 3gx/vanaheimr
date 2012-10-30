/*! \file   CoreSimKernel.h
	\date   Sunday September 19, 2011
	\author Sudnya Padalikar
		<mailsudnya@gmail.com>
	\brief  The header file for the Core simulator of the Kernel class.
*/

#pragma once

/*! \brief A namespace for program execution */
//forward declarations
namespace archaeopteryx { namespace executive { class CoreSimBlock; } }
namespace archaeopteryx { namespace        ir { class Binary;       } }

namespace archaeopteryx
{

namespace executive
{

class CoreSimKernel
{
    public:
        //__device__ CoreSimKernel(void *gpuState, char* binaryName);
        __device__ void launchKernel(unsigned int simulatedBlocks,
			executive::CoreSimBlock* blocks, ir::Binary* binary);
        
    public:
    	// Interface to CoreSimBlock
    	__device__ size_t translateVirtualToPhysicalAddress(
    		size_t virtualAddress) const;

};

}

}

