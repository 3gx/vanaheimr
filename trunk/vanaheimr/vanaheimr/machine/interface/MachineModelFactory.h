/*! \file   MachineModelFactory.h
	\date   Tuesday January 15, 2013
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the MachineModelFactory class.
	
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/Analysis.h>

namespace vanaheimr
{

namespace analysis
{

/*! \brief Used to create passes by name */
class MachineModelFactory
{
public:
	typedef MachineModel::StringVector StringVector;

public:
	/*! \brief Create a machine model object from the specified name */
	static Analysis* createMachineModel(const std::string& name,
		const StringVector& options = StringVector());

};

}

}

