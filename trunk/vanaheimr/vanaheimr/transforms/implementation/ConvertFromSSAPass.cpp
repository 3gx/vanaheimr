/*! \file   ConvertFromSSAPass.cpp
	\date   Tuesday November 20, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the ConvertFromSSAPass class.
*/

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/ConvertFromSSAPass.h>

#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/Instruction.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

namespace vanaheimr
{

namespace transforms
{

ConvertFromSSAPass::ConvertFromSSAPass()
: FunctionPass(StringVector({"ControlFlowGraph", "DataflowAnalysis"}),
  "ConvertFromSSAPass")
{
	
}

void ConvertFromSSAPass::runOnFunction(Function& f)
{
	_removePhis(f);
	_removePsis(f);
}

void ConvertFromSSAPass::_removePhis(Function& f)
{
	// for all
	for(auto block = f.begin(); block != f.end(); ++block)
	{
		_removePhis(*block);
	}
}

void ConvertFromSSAPass::_removePsis(Function& f)
{
	// for all
	for(auto block = f.begin(); block != f.end(); ++block)
	{
		// for all?
		for(auto instruction : *block)
		{
			if(!instruction->isPsi()) continue;
			
			_removePsi(static_cast<ir::Psi&>(*instruction));
		}
	}
}

void ConvertFromSSAPass::_removePhis(ir::BasicBlock& block)
{
	// get the first instruction after the phi
	//auto phiEnd = getFirstNonPhiInstruction(block);
	
	// 
}

void ConvertFromSSAPass::_removePsi(ir::Psi& psi)
{
	assertM(false, "Not implemented.");
}


}

}


