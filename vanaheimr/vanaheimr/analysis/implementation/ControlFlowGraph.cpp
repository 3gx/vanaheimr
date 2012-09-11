/*! \file   ControlFlowGraph.cpp
	\date   Monday September 10, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the control flow graph class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/analysis/includes/ControlFlowGraph.h>

namespace vanaheimr
{

namespace analysis
{

ControlFlowGraph::ControlFlowGraph()
: FunctionAnalysis("ControlFlowGraph")
{

}

BasicBlock* ControlFlowGraph::getEntryBlock()
{
	return _entry;
}

BasicBlock* ControlFlowGraph::getExitBlock()
{
	return _exit;
}

const BasicBlock* ControlFlowGraph::getEntryBlock() const
{
	return _entry;
}

const BasicBlock* ControlFlowGraph::getExitBlock() const
{
	return _exit;
}

BasicBlockSet ControlFlowGraph::getSuccessors(const BasicBlock& b)
{
	return _successors[];
}

BasicBlockSet ControlFlowGraph::getPredecessors(const BasicBlock& b);

bool ControlFlowGraph::isEdge(const BasicBlock& head, const BasicBlock& edge);
bool ControlFlowGraph::isBranchEdge(const BasicBlock& head, const BasicBlock& edge);
bool ControlFlowGraph::isFallthroughEdge(const BasicBlock& head, const BasicBlock& edge);

Function* ControlFlowGraph::function()
{
	
}

const Function* ControlFlowGraph::function() const
{

}

void ControlFlowGraph::analyze(Function& function)
{
	assertM(false, "Not implemented.");
}

}


}

