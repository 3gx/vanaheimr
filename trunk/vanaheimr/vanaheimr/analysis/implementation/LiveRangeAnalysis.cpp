/*! \file   LiveRangeAnalysis.cpp
	\date   Saturday January 12, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the live-range analysis class.
*/

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/LiveRangeAnalysis.h>

#include <vanaheimr/analysis/interface/DataflowAnalysis.h>
#include <vanaheimr/analysis/interface/ControlFlowGraph.h>

#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/VirtualRegister.h>

// Standard Library Includes
#include <cassert>

namespace vanaheimr
{

namespace analysis
{

typedef LiveRangeAnalysis::LiveRange LiveRange;

LiveRange::LiveRange(LiveRangeAnalysis* liveRangeAnalysis, VirtualRegister* vr)
: _analysis(liveRangeAnalysis), _virtualRegister(vr)
{

}

LiveRangeAnalysis* LiveRangeAnalysis::LiveRange::liveRangeAnalysis() const
{
	return _analysis;
}

ir::VirtualRegister* LiveRangeAnalysis::LiveRange::virtualRegister() const
{
	return _virtualRegister;
}

LiveRangeAnalysis::LiveRangeAnalysis()
: FunctionAnalysis("LiveRangeAnalysis", {"DataflowAnalysis", "ControlFlowGraph"})
{

}

const LiveRange* LiveRangeAnalysis::getLiveRange(
	const VirtualRegister& virtualRegister) const
{
	assert(virtualRegister.id < _liveRanges.size());

	return &_liveRanges[virtualRegister.id];
}

LiveRangeAnalysis::LiveRange* LiveRangeAnalysis::getLiveRange(
	const VirtualRegister& virtualRegister)
{
	assert(virtualRegister.id < _liveRanges.size());

	return &_liveRanges[virtualRegister.id];
}

static void findLiveRange(LiveRangeAnalysis::LiveRange& liveRange,
	DataflowAnalysis* dfg, ControlFlowGraph* );

void LiveRangeAnalysis::analyze(Function& function)
{
	auto dfg = static_cast<DataflowAnalysis*>(getAnalysis("DataflowAnalysis"));
	assert(dfg != nullptr);

	auto cfg = static_cast<ControlFlowGraph*>(getAnalysis("ControlFlowGraph"));
	assert(cfg != nullptr);

	_initializeLiveRanges(function);

	// compute the live range for each variable in parallel (for all)
	// TODO: use an algorithm that merges partial results
	for(auto virtualRegister = function.register_begin();
		virtualRegister != function.register_end(); ++virtualRegister)
	{
		findLiveRange(*getLiveRange(*virtualRegister), dfg, cfg);
	}
}

void LiveRangeAnalysis::_initializeLiveRanges(Function& function)
{
	_liveRanges.clear();
	_liveRanges.reserve(function.register_size());

	for(auto virtualRegister = function.register_begin();
		virtualRegister != function.register_end(); ++virtualRegister)
	{
		_liveRanges.push_back(LiveRange(this, &*virtualRegister));
	}
}

typedef ir::BasicBlock BasicBlock;

static bool isLiveOut(LiveRange& liveRange, BasicBlock* block,
	DataflowAnalysis* dfg)
{
	auto liveOuts = dfg->getLiveOuts(*block);

	return liveOuts.count(liveRange.virtualRegister()) != 0;
}

static bool blockHasDefinitions(BasicBlock* block, const LiveRange& liveRange)
{
	for(auto definition : liveRange.definingInstructions)
	{
		if(definition->block == block) return true;
	}

	return false;
}

typedef LiveRangeAnalysis::BasicBlockSet BasicBlockSet;

static void walkUpPredecessor(BasicBlockSet& visited, LiveRange& liveRange,
	BasicBlock* block, DataflowAnalysis* dfg, ControlFlowGraph* cfg)
{
	// early exit when a node is already visited
	if(!visited.insert(block).second) return;

	// skip nodes for which the value is not live-out
	if(!isLiveOut(liveRange, block, dfg)) return;

	// skip nodes that define the value, starting the live range
	if(blockHasDefinitions(block, liveRange)) return;

	// add the block to the live range
	liveRange.fullyUsedBlocks.insert(block);

	// recurse on predecessors with the value as a live out
	auto predecessors = cfg->getPredecessors(*block);
	
	for(auto predecessor : predecessors)
	{
		walkUpPredecessor(visited, liveRange, predecessor, dfg, cfg);
	}
	
}

typedef ir::Instruction Instruction;

static bool blockHasPriorDefinitions(LiveRange& liveRange, Instruction* user)
{
	for(auto definition : liveRange.definingInstructions)
	{
		// Is there a definition in the same block
		if(definition->block != user->block) continue;
		
		// Does the definition occur prior to this use?
		if(definition->index() < user->index()) return true;
	}

	return false;
}

static void walkUpDataflowGraph(LiveRange& liveRange,
	DataflowAnalysis* dfg, ControlFlowGraph* cfg, Instruction* user)
{
	BasicBlockSet visited;

	visited.insert(user->block);

	// skip blocks that start the live range
	if(blockHasPriorDefinitions(liveRange, user)) return;

	auto predecessors = cfg->getPredecessors(*user->block);
	
	for(auto predecessor : predecessors)
	{
		walkUpPredecessor(visited, liveRange, predecessor, dfg, cfg);
	}
}

static void findLiveRange(LiveRangeAnalysis::LiveRange& liveRange,
	DataflowAnalysis* dfg, ControlFlowGraph* cfg)
{
	auto vr = liveRange.virtualRegister();

	liveRange.definingInstructions = dfg->getReachingDefinitions(*vr);
	liveRange.usingInstructions    = dfg->getReachedUses(*vr);

	// in parallel
	for(auto use : liveRange.usingInstructions)
	{
		walkUpDataflowGraph(liveRange, dfg, cfg, use);
	}
}

}

}

