/*! \file   DataflowAnalysis.h
	\date   Monday September 10, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the dataflow analysis class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/analysis/includes/Analysis.h>

#include <vanaheimr/util/includes/SmallSet.h>

// Forward Declarations
namespace vanaheimr { namespace ir { class VirtualRegister; } }

namespace vanaheimr
{

namespace analysis
{

/*! \brief A class for performing dataflow analysis */	
class DataflowAnalysis : public FunctionAnalysis
{
public:
	typedef util::SmallSet<VirtualRegister*> VirtualRegisterSet;
	typedef util::SmallSet<Instruction*>     InstructionSet;

public:
	VirtualRegisterSet  getLiveIns(const BasicBlock&);
	VirtualRegisterSet getLiveOuts(const BasicBlock&);

public:
	InstructionSet getReachingDefs(const Instruction&);
	InstructionSet getReachedUsers(const Instruction&);

public:
	virtual void analyze(Function& function);
};

}

}


