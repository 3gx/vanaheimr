/*! \file   ReversePostOrderTraversal.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Tuesday June 23, 2012
	\file   The source file for the ReversePostOrderTraversal class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/ReversePostOrderTraversal.h>

namespace vanaheimr
{

namespace analysis
{

ReversePostOrderTraversal::ReversePostOrderTraversal()
: FunctionAnalysis("ReversePostOrderTraversal",
  StringVector(1, "ControlFlowGraph"))
{

}

void ReversePostOrderTraversal::analyze(Function& function)
{
	order.clear();
	order.resize(function.size());
	
	BlockSet   visited;
	BlockStack stack;
	
	auto cfg = getAnalysis<ControlFlowGraph>("ControlFlowGraph");
	
	// Post order is left subtree, right subtree, current node
	// Parallelizing this will be tricky
	stack.push(function.entry_block());
	
	while(!stack.empty())
	{
		BasicBlock* top = stack.top();
		
		auto successors = cfg->getSuccessors(*top);
		
		bool allSuccessorsVisited = true;
		
		for(auto successor : successors)
		{
			if(visited.insert(successor).second)
			{
				allSuccessorsVisited = false;
				stack.push(successor);
			}
		}
		
		if(!allSuccessorsVisited) continue;
		
		stack.pop();
		order.push_back(top);
	}
	
	// reverse the order
	std::reverse(order.begin(), order.end());
}

}

}



