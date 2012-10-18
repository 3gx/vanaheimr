/*! \file   ReversePostOrderTraversal.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Tuesday June 23, 2012
	\file   The source file for the ReversePostOrderTraversal class.
*/

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/ReversePostOrderTraversal.h>
#include <vanaheimr/analysis/interface/ControlFlowGraph.h>

#include <vanaheimr/ir/interface/Function.h>

#include <vanaheimr/util/interface/LargeSet.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Include
#include <stack>
#include <algorithm>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

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
	typedef util::LargeSet<BasicBlock*> BlockSet;
	typedef std::stack<BasicBlock*>     BlockStack;

	order.clear();
	
	BlockSet   visited;
	BlockStack stack;
	
	auto cfgAnalysis = getAnalysis("ControlFlowGraph");
	auto cfg         = static_cast<ControlFlowGraph*>(cfgAnalysis);	

	report("Creating reverse post order traversal over function '" +
		function.name() + "'");

	// Post order is left subtree, right subtree, current node
	// Parallelizing this will be tricky
	stack.push(&*function.entry_block());
	
	while(!stack.empty())
	{
		BasicBlock* top = stack.top();
		
		auto successors = cfg->getSuccessors(*top);
		
		bool allSuccessorsVisited = true;
		
		for(auto successor : successors)
		{
			assert(successor != nullptr);
			if(visited.insert(successor).second)
			{
				allSuccessorsVisited = false;
				stack.push(successor);
			}
		}
		
		if(!allSuccessorsVisited) continue;
		
		stack.pop();
		order.push_back(top);
		
		report(" BB_" << top->id());
	}
	
	// reverse the order
	std::reverse(order.begin(), order.end());
}

}

}



