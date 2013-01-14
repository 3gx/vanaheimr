/*! \file   LiveRangeAnalysis.h
	\date   Saturday January 12, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the live-range analysis class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/Analysis.h>

#include <vanaheimr/util/interface/SmallSet.h>

// Forward Declarations
namespace vanaheimr { namespace ir { class VirtualRegister;  } }
namespace vanaheimr { namespace ir { class Instruction;      } }
namespace vanaheimr { namespace ir { class BasicBlock;       } }

namespace vanaheimr
{

namespace analysis
{

/*! \brief A class for performing live range analysis */	
class LiveRangeAnalysis : public FunctionAnalysis
{
public:
	typedef      ir::BasicBlock BasicBlock;
	typedef     ir::Instruction Instruction;
	typedef ir::VirtualRegister VirtualRegister;

	typedef util::SmallSet<Instruction*> InstructionSet;
	typedef util::SmallSet<BasicBlock*>  BasicBlockSet;

	class LiveRange
	{
	public:
		LiveRange(LiveRangeAnalysis*, VirtualRegister*);
	
	public:
		LiveRangeAnalysis* liveRangeAnalysis() const;
		  VirtualRegister*   virtualRegister() const;

	public:
		BasicBlockSet fullyUsedBlocks;

	public:
		InstructionSet definingInstructions;
		InstructionSet usingInstructions;	

	private:
		LiveRangeAnalysis* _analysis;
		VirtualRegister*   _virtualRegister;
	};


public:
	LiveRangeAnalysis();
	
public:
	const LiveRange* getLiveRange(const VirtualRegister&) const;
	      LiveRange* getLiveRange(const VirtualRegister&);
	
public:
	virtual void analyze(Function& function);

public:
	LiveRangeAnalysis(const LiveRangeAnalysis& ) = delete;
	LiveRangeAnalysis& operator=(const LiveRangeAnalysis& ) = delete;
	
private:
	typedef std::vector<LiveRange> LiveRangeVector;

private:
	void _initializeLiveRanges(ir::Function& );

private:
	LiveRangeVector _liveRanges;

};

}

}



