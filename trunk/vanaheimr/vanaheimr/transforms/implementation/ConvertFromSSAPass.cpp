/*! \file   ConvertFromSSAPass.cpp
	\date   Tuesday November 20, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the ConvertFromSSAPass class.
*/

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/ConvertFromSSAPass.h>

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
			
			_removePsi(static_cast<ir::Phi&>(*instruction));
		}
	}
}

void ConvertFromSSAPass::removePhis(ir::BasicBlock& block)
{
	// get the first instruction after the phi
	auto phiEnd = getFirstNonPhiInstruction(block);
	
	// 
}

void ConvertFromSSAPass::removePsi(ir::Psi& psi)
{
	assertM(false, "Not implemented.");
}


}

}


