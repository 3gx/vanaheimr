/*! \file   ArchaeopteryxTarget.h
	\date   Thursday December 20, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the ArchaeopteryxTarget class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/Target.h>

namespace vanaheimr
{

namespace codegen
{

/*! \brief A target for the Archaeopteryx Simulator */
class ArchaeopteryxTarget : public Target
{
public:
	ArchaeopteryxTarget();

public:
	/*! \brief Lower the assigned module to the target ISA */
	void lower();
	/*! \brief Get lowered module in the target ISA */
	ir::ModuleBase* getLoweredModule();

public:
	std::string registerAllocatorName;
	std::string instructionSchedulerName;

};

}

}


