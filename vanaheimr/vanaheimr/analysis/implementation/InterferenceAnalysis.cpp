/*! \file   InterferenceAnalysis.cpp
	\date   Saturday January 12, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the InterferenceAnalysis class.
*/

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/InterferenceAnalysis.h>

namespace vanaheimr
{

namespace analysis
{

InterferenceAnalysis::InterferenceAnalysis()
: FunctionAnalysis("InterferenceAnalysis", {"LiveRangeAnalysis"})
{

}

bool InterferenceAnalysis::doLiveRangesInterfere(const VirtualRegister& one,
	const VirtualRegister& two) const
{
	auto interferences = getInterferences(one);

	return interferences.count(&two) != 0;
}

InterferenceAnalysis::VirtualRegisterSet&
	InterferenceAnalysis::getInterferences(
		const VirtualRegister& virtualRegister)
{
	assert(virtualRegister.id < _interferences.size());

	return _interferences[virtualRegister.id];
}

const InterferenceAnalysis::VirtualRegisterSet&
	InterferenceAnalysis::getIntereference(
		const VirtualRegister& virtualRegister) const
{
	assert(virtualRegister.id < _interferences.size());

	return _interferences[virtualRegister.id];
}

void InterferenceAnalysis::analyze(Function& function)
{
	typedef std::pair<ir::BasicBlock*, LiveRange*> BlockToRange;
	typedef std::vector<BlockToRange> BlockToRangeVector;

	// Determine interferences among fully covered blocks
	BlockToRangeVector blocksToRanges;

	std::sort(blocksToRanges.begin(), blocksToRanges.end());

	// Determine interferences 
}

}

}

