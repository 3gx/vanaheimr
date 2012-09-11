/*! \file   DominatorAnalysis.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Tuesday June 23, 2009
	\file   The header file for the DominatorAnalysis class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/analysis/includes/Analysis.h>

namespace vanaheimr
{

namespace analysis
{

class DominatorAnalysis : public FunctionAnalysis
{
public:
	typedef util::SmallSet<BasicBlock*> BasicBlockSet;
	
public:
	/*! \brief Is a block dominated by another? */
	bool dominates(const BasicBlock& b, const BasicBlock& potentialDominator);

	/*! \brief Find the immediate dominator of a given block */
	BasicBlock* getDominator(const BasicBlock& b);
	
	/*! \brief Get the set of blocks immediately dominated by this block */
	BlockSet getDominatedBlocks(const BasicBlock& b);
	
public:
	virtual void analyze(Function& function);

};

}

}


