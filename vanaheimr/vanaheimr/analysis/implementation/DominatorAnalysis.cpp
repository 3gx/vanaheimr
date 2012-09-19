/*! \file   DominatorAnalysis.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Tuesday September 18, 2012
	\file   The source file for the DominatorAnalysis class.
*/

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/DominatorAnalysis.h>

#include <vanaheimr/analysis/interface/ControlFlowGraph.h>
#include <vanaheimr/analysis/interface/ReversePostOrderTraversal.h>

#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/BasicBlock.h>

// Standard Library Includes
#include <cassert>

namespace vanaheimr
{

namespace analysis
{

DominatorAnalysis::DominatorAnalysis()
: FunctionAnalysis("DominatorAnalysis", StringVector(1, "ControlFlowGraph"))
{

}


bool DominatorAnalysis::dominates(const BasicBlock& b,
	const BasicBlock& potentialDominator)
{
	auto dominatedBlocks = getDominatedBlocks(potentialDominator);

	return dominatedBlocks.count(const_cast<BasicBlock*>(&b)) != 0;
}

DominatorAnalysis::BasicBlock* DominatorAnalysis::getDominator(
	const BasicBlock& b)
{
	assert(b.id() < _immediateDominators.size());
	return _immediateDominators[b.id()];
}

const DominatorAnalysis::BasicBlockSet&
	DominatorAnalysis::getDominatedBlocks(const BasicBlock& b)
{
	assert(b.id() < _dominatedBlocks.size());
	return _dominatedBlocks[b.id()];
}

typedef std::vector<unsigned int> IntVector; 

ir::BasicBlock* intersect(DominatorAnalysis* tree,
	const IntVector& postOrderNumbers,
	ir::BasicBlock* left, ir::BasicBlock* right)
{
	auto finger1 = left;
	auto finger2 = right;
	while(postOrderNumbers[finger1->id()] != postOrderNumbers[finger2->id()])
	{
		while(postOrderNumbers[finger1->id()] < postOrderNumbers[finger2->id()])
		{
			finger1 = tree->getDominator(*finger1);
		}
		while(postOrderNumbers[finger2->id()] < postOrderNumbers[finger1->id()])
		{
			finger2 = tree->getDominator(*finger2);
		}
	}
	
	return finger1;
}

void DominatorAnalysis::analyze(Function& function)
{
	_immediateDominators.clear();
	
	// Get dependent analyses
	auto cfg              = static_cast<ControlFlowGraph*>(
		getAnalysis("ControlFlowGraph"));
	auto reversePostOrder = static_cast<ReversePostOrderTraversal*>(
		getAnalysis("ReversePostOrderTraversal"));
	
	// Determine post order numbers
	IntVector postOrderNumbers(function.size());
	
	for(auto block = reversePostOrder->order.begin();
		block != reversePostOrder->order.end(); ++block)
	{
		postOrderNumbers[(*block)->id()] =
			std::distance(block, reversePostOrder->order.begin());
	}
	
	// All blocks start being uninitialized
	_immediateDominators.assign(function.size(), nullptr);
	
	// The entry starts dominating itself
	_immediateDominators[function.entry_block()->id()] =
		&*function.entry_block();
	
	// Propagate changes, serial for each iteration
	bool changed = true;
	
	while(changed)
	{
		changed = false;
	
		// Run over blocks in reverse post order
		// TODO, can this be done in parallel?
		for(auto block : reversePostOrder->order)
		{
			// Get all predecessors
			auto predecessors = cfg->getPredecessors(*block);
		
			if(predecessors.empty()) continue;
			
			// The new dominator is the first predecessor
			BasicBlock* newDominator = nullptr;
			
			for(auto predecessor : predecessors)
			{
				if(getDominator(*predecessor) == nullptr) continue;
				
				if(newDominator == nullptr)
				{
					newDominator = getDominator(*predecessor);
					continue;
				}
				
				newDominator = intersect(this, postOrderNumbers,
					newDominator, predecessor);
			}
			
			if(newDominator != getDominator(*block))
			{
				_immediateDominators[block->id()] = newDominator;
			}
		}
	}

	_dominatedBlocks.resize(function.size());
	
	// Update the dominated set, 
	//  This is another reverse insert operation
	//   we can use atomics or sort+group_by_key for a parallel implementation
	for(auto block = function.begin(); block != function.end(); ++block)
	{
		_dominatedBlocks[getDominator(*block)->id()].insert(&*block);
	}
}

}

}


