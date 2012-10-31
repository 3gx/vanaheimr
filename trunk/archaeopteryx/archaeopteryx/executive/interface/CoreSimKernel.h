/*! \file   CoreSimKernel.h
	\date   Sunday September 19, 2011
	\author Sudnya Padalikar
		<mailsudnya@gmail.com>
	\brief  The header file for the Core simulator of the Kernel class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/util/interface/IntTypes.h>

// Forward declarations
namespace archaeopteryx { namespace executive { class CoreSimBlock; } }
namespace archaeopteryx { namespace        ir { class Binary;       } }

namespace archaeopteryx
{

namespace executive
{

class CoreSimKernel
{
public:
	typedef uint64_t Address;

public:
   __device__ void launchKernel(unsigned int simulatedBlocks,
		executive::CoreSimBlock* blocks, ir::Binary* binary);
    
public:
	// Interface to CoreSimBlock
	__device__ Address translateVirtualToPhysicalAddress(
		Address virtualAddress) const;

};

}

}

