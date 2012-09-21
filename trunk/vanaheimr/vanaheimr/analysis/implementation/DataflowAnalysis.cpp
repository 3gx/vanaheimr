/*! \file   DataflowAnalysis.cpp
	\date   Friday September 14, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the dataflow analysis class.
*/

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/DataflowAnalysis.h>
#include <vanaheimr/analysis/interface/ControlFlowGraph.h>

#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/BasicBlock.h>

namespace vanaheimr
{

namespace analysis
{

DataflowAnalysis::DataflowAnalysis()
: FunctionAnalysis("DataflowAnalysis", StringVector(1, "ControlFlowGraph"))
{

}

DataflowAnalysis::VirtualRegisterSet
	DataflowAnalysis::getLiveIns(const BasicBlock& block)
{
	return _liveins[block.id()];
}

DataflowAnalysis::VirtualRegisterSet
	DataflowAnalysis::getLiveOuts(const BasicBlock& block)
{
	return _liveouts[block.id()];
}

DataflowAnalysis::InstructionSet
	DataflowAnalysis::getReachingDefinitions(const Instruction& instruction)
{
	return _reachingDefinitions[instruction.id()];
}

DataflowAnalysis::InstructionSet 
	DataflowAnalysis::getReachedUses(const Instruction& instruction)
{
	return _reachedUses[instruction.id()];
}

void DataflowAnalysis::analyze(Function& function)
{
	     _analyzeLiveInsAndOuts(function);
	_analyzeReachingDefinitions(function);
}

void DataflowAnalysis::_analyzeLiveInsAndOuts(Function& function)
{
	 _liveins.resize(function.size());
	_liveouts.resize(function.size());
	
	BasicBlockSet worklist;
	
	// should be for-all
	for(auto block = function.begin(); block != function.end(); ++block)
	{
		worklist.insert(&*block);
	}
	
	while(!worklist.empty())
	{
		_computeLocalLiveInsAndOuts(worklist);
	}
}

void DataflowAnalysis::_analyzeReachingDefinitions(Function& function)
{
	// TODO implement this
}

void DataflowAnalysis::_computeLocalLiveInsAndOuts(BasicBlockSet& worklist)
{
	BasicBlockSet newList;

	// should be for-all
	for(auto block : worklist)
	{
		bool changed = _recomputeLiveInsAndOutsForBlock(block);

		if(changed)
		{
			// TODO: queue up predecessors
			newList.insert(block);
		}
	}

	// gather blocks to form the new worklist
	worklist = std::move(newList);
}	

bool DataflowAnalysis::_recomputeLiveInsAndOutsForBlock(BasicBlock* block)
{
	// live outs is the union of live-ins of all successors
	VirtualRegisterSet liveout;

	auto successors = _cfg->getSuccessors(*block);

	for(auto successor : successors)
	{
		auto livein = getLiveIns(*successor);

		liveout.insert(livein.begin(), livein.end());
	}

	_liveouts[block->id()] = liveout;

	VirtualRegisterSet livein = std::move(liveout);

	// apply def/use rules in reverse order
	for(auto instruction = block->rbegin(); instruction != block->rend();
		++instruction)
	{
		// spawn on uses
		for(auto read : (*instruction)->reads)
		{
			if(!read->isRegister()) continue;
		
			auto reg = static_cast<ir::RegisterOperand*>(read);

			livein.insert(reg->virtualRegister);
		}

		// kill on defs
		for(auto write : (*instruction)->writes)
		{
			if(!write->isRegister()) continue;
		
			auto reg = static_cast<ir::RegisterOperand*>(write);

			livein.erase(reg->virtualRegister);
		}
	}

	bool changed = _liveins[block->id()] == livein;

	_liveins[block->id()] = std::move(livein);
	
	return changed;
}

}

}


