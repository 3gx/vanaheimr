/* 	\file Intrinsics.cpp
	\date Tuesday February 4, 2013
	\author Gregory Diamos
	\brief The source file for the archeopteryx intrinsic functions.

*/

// Archaeopteryx Includes
#include <archaeopteryx/executive/interface/Intrinsics.h>
#include <archaeopteryx/executive/interface/CoreSimBlock.h>

#include <archaeopteryx/ir/interface/Binary.h>

#include <archaeopteryx/util/interface/debug.h>
#include <archaeopteryx/util/interface/string.h>

// Vanaheimr Includes
#include <vanaheimr/asm/interface/Operand.h>
#include <vanaheimr/asm/interface/Instruction.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace archaeopteryx
{

namespace executive
{

__device__ bool Intrinsics::isIntrinsic(const vanaheimr::as::Call* call,
	CoreSimBlock* block)
{
	cta_report("Checking if call is an intrinsic.\n");

	if(call->target.asOperand.mode != vanaheimr::as::Operand::Symbol)
	{
		return false;
	}

	const vanaheimr::as::SymbolOperand* symbol = &call->target.asSymbol;

	cta_report(" checking if symbol '%d' is "
		"an intrinsisc...\n", symbol->symbolTableOffset);

	util::string name = block->binary()->getSymbolName(symbol->symbolTableOffset);

	bool isIntrinsic = name.find("_Zintrinsic") == 0;

	if(isIntrinsic)
	{
		cta_report("  it is\n");
	}
	else
	{
		cta_report("  it isn't\n");
	}

	return isIntrinsic;
}

__device__ void Intrinsics::execute(const vanaheimr::as::Call* call,
	CoreSimBlock* block, unsigned int threadId)
{
	device_assert(false);

}

}

}

