/*	\file   Operand.h
	\date   Tuesday February 5, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for Operand accessor functions.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/asm/interface/Operand.h>

// Forward Declarations
namespace archaeopteryx { namespace executive { class CoreSimBlock; } }

namespace archaeopteryx
{

namespace executive
{

__device__ uint64_t getOperand(
	const vanaheimr::as::OperandContainer& operandContainer,
	CoreSimBlock* parentBlock, unsigned threadId);

__device__ uint64_t getOperand(const vanaheimr::as::Operand& operand,
	CoreSimBlock* parentBlock, unsigned threadId);

}

}


