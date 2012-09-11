/*! \file   ControlFlowGraph.h
	\date   Monday September 10, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the control flow graph class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/analysis/includes/Analysis.h>

#include <vanaheimr/util/includes/SmallSet.h>

namespace vanaheimr
{

namespace analysis
{

class ControlFlowGraph : public FunctionAnalysis
{
public:
	typedef util::SmallSet<BasicBlock*> BasicBlockSet;

public:
	ControlFlowGraph();

public:
	BasicBlock* getEntryBlock();
	BasicBlock* getExitBlock();
	
	const BasicBlock* getEntryBlock() const;
	const BasicBlock* getExitBlock()  const;

public:
	BasicBlockSet   getSuccessors(const BasicBlock&);
	BasicBlockSet getPredecessors(const BasicBlock&);

public:
	bool            isEdge(const BasicBlock& head, const BasicBlock& edge);
	bool      isBranchEdge(const BasicBlock& head, const BasicBlock& edge);
	bool isFallthroughEdge(const BasicBlock& head, const BasicBlock& edge);

public:
	      Function* function();
	const Function* function() const;

public:
	virtual void analyze(Function& function);

private:
	Function* _function;
};

}


}

