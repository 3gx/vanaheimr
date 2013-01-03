/*! \file   ListInstructionSchedulerPass.cpp
	\date   Sunday December 23, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the ListInstructionSchedulerPass class.
*/

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/ListInstructionSchedulerPass.h>

#include <vanaheimr/analysis/interface/DataflowAnalysis.h>

#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/BasicBlock.h>

#include <vanaheimr/util/interface/LargeSet.h>

// Standard Library Includes
#include <cassert>

namespace vanaheimr
{

namespace codegen
{

ListInstructionSchedulerPass::ListInstructionSchedulerPass()
: FunctionPass({"DataflowAnalysis"}, "ListInstructionSchedulerPass")
{

}

typedef util::LargeSet<ir::Instruction*> InstructionSet;
	
static bool anyDependencies(ir::Instruction* instruction,
	analysis::DataflowAnalysis& dfg,
	const InstructionSet& remainingInstructions)
{
	auto reachingDefiningInstructions =
			dfg.getReachingDefinitions(*instruction);

	bool anyDependencies = !reachingDefiningInstructions.empty();
		
	for(auto writer : reachingDefiningInstructions)
	{
		if(writer->block != instruction->block) continue;
		
		if(writer->id() < instruction->id())
		{
			anyDependencies = true;
			break;
		}
	}
	
	return anyDependencies;
}

static void schedule(ir::BasicBlock& block, analysis::DataflowAnalysis& dfg)
{
	// TODO sort by priority, sort in parallel
	ir::BasicBlock::InstructionList newInstructions;
	
	InstructionSet readyInstructions;
	
	InstructionSet remainingInstructions;
	
	remainingInstructions.insert(block.begin(), block.end());

	for(auto instruction : block)
	{
		if(!anyDependencies(instruction, dfg, remainingInstructions))
		{
			readyInstructions.insert(instruction);
		}
	}

	while(!readyInstructions.empty())
	{
		auto next = *readyInstructions.begin();
		readyInstructions.erase(readyInstructions.begin());

		newInstructions.push_back(next);

		// free dependent instructions
		auto reachedUses = dfg.getReachedUses(*next);

		for(auto use : reachedUses)
		{
			if(!anyDependencies(use, dfg, remainingInstructions))
			{
				readyInstructions.insert(use);
			}
		}
	}

	assert(newInstructions.size() == block.size());

	block.assign(newInstructions.begin(), newInstructions.end());
}

void ListInstructionSchedulerPass::runOnFunction(Function& f)
{
	auto dfg = static_cast<analysis::DataflowAnalysis*>(
		getAnalysis("DataflowAnalysis"));
	
	// for all blocks
	for(auto block = f.begin(); block != f.end(); ++block)
	{
		schedule(*block, *dfg);
	}
}

}

}



