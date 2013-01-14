/*! \file   InterferenceAnalysis.cpp
	\date   Saturday January 12, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the InterferenceAnalysis class.
*/

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/InterferenceAnalysis.h>

#include <vanaheimr/analysis/interface/LiveRangeAnalysis.h>

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
	const VirtualRegisterSet& interferences = getInterferences(one);

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
	InterferenceAnalysis::getInterferences(
		const VirtualRegister& virtualRegister) const
{
	assert(virtualRegister.id < _interferences.size());

	return _interferences[virtualRegister.id];
}

typedef std::pair<ir::BasicBlock*, LiveRange*> BlockToRange;
typedef std::vector<BlockToRange> BlockToRangeVector;
typedef BlockToRangeVector::iterator RangeIterator;
typedef std::pair<RangeIterator, RangeIterator> Range;
typedef std::vector<Range> RangeVector;

static BlockToRangeVector mapBlocksToLiveRanges(LiveRangeAnalysis*);
static RangeVector partition(BlockToRangeVector&);

void InterferenceAnalysis::analyze(Function& function)
{
	auto ranges = static_cast<LiveRangeAnalysis*>(
		getAnalysis("LiveRangeAnalysis"));
	assert(ranges != nullptr);

	// partition into ranges with equal blocks
	auto blocksToRanges = mapBlocksToLiveRanges(ranges);

	auto partitions = partition(blocksToRanges);

	// check all partitions (TODO in parallel)
	for(auto partition : partitions)
	{
		for(auto one = partition.first; one != partition.second; ++one)
		{
			for(auto two = partition.first; two != partition.second; ++two)
			{
				if(one == two) continue;

				if(one->second->intersect(*two->second))
				{
					getInterference(*one->second->virtualRegister()).insert(
						two->second->virtualRegister());
				}
			}
		}
	}
}
	


	// Determine group live ranges by blocks that they reference 
	BlockToRangeVector blocksToRanges;

	std::sort(blocksToRanges.begin(), blocksToRanges.end());


}

}

