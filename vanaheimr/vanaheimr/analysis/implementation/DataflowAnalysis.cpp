/*! \file   DataflowAnalysis.cpp
	\date   Friday September 14, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the dataflow analysis class.
*/

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/DataflowAnalysis.h>

namespace vanaheimr
{

namespace analysis
{

DataflowAnalysis::DataflowAnalysis()
: FunctionAnalysis("DataflowAnalysis", StringVector("ControlFlowGraph"))
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
		worklist.insert(block);
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
	

}

}


