/*! \file   ListInstructionSchedulerPass.cpp
	\date   Sunday December 23, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the ListInstructionSchedulerPass class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/ListInstructionSchedulerPass.h>

namespace vanaheimr
{

namespace transforms
{

ListInstructionSchedulerPass::ListInstructionSchedulerPass()
: FunctionPass({"DataflowAnalysis"}, "ListInstructionSchedulerPass")
{

}

static void schedule(ir::BasicBlock& block, dfg)
{
	// TODO sort by priority, sort in parallel
	typedef LargeSet<ir::Instruction*> InstructionSet;
	
	ir::BasicBlock::InstructionList newInstructions;
	
	InstructionSet readyInstructions;
	
	InstructionSet remainingInstructions(block.begin(), block.end());

	for(auto instruction : block)
	{
		auto reachingDefiningInstructions =
			dfg.getReachingDefinitions(instruction);
		
		bool anyDependencies = !reachingDefiningInstructions.empty();
		
		for(auto writer : reachingDefiningInstructions)
		{
			if(writer->block != &block) continue;
			
			if(write->id < instruction->id)
			{
				anyDependencies = true;
				break;
			}
		}

		if(!anyDependencies(instruction, dfg, remainingInstructions))
		{
			readyInstructions.insert(instruction);
		}
	}

	while(!readyInstructions.empty())
	{
		auto next = *readyInstructions.begin();
		readyInstruction.erase(readyInstructions.begin());

		readyInstructions.push_back(next);

		// free dependent instructions
		auto reachedUses = dfg.getReachedUses(next);

		for(auto use : reachedUses)
		{
			if(!anyDependencies(use, dfg, remainingInstructions))
			{
				readyInstructions.insert(use);
			}
		}
	}

	assert(newInstructions.size() == block.size());

	block.instructions = newInstructions;

	// renumber
	block.renumberInstructions();
}

void ListInstructionSchedulerPass::runOnFunction(Function& f)
{
	// for all blocks
	for(auto block = f.begin(); block != f.end(); ++block)
	{
		schedule(*block);
	}
}

}

}



