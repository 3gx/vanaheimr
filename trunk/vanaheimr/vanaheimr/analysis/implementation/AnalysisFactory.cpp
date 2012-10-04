/*! \file   AnalysisFactory.cpp
	\date   Wednesday October 3, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The source file for the AnalysisFactory class.
	
*/

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/AnalysisFactory.h>

namespace vanaheimr
{

namespace analysis 
{

Analysis* AnalysisFactory::createAnalysis(const std::string& name,
	const StringVector& options)
{
	Analysis* analysis = nullptr;

	if(analysis != nullptr)
	{
		analysis->configure(options);
	}
	
	return analysis;
}

}

}

