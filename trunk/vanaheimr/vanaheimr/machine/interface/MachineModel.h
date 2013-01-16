/*! \file   MachineModel.h
	\date   Thursday January 3, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the MachineModel class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

// Forward Declarations
namespace vanaheimr { namespace machine { class PhysicalRegister; } }

namespace vanaheimr
{

/*! \brief A namespace for the Vanaheimr machine model */
namespace machine
{

/*! \brief A model of a vanaheimr processor */
class MachineModel
{
public:
	typedef unsigned int             RegisterId;
	typedef std::vector<std::string> StringVector;

public:
	/*! \brief Construct a machine model */
	MachineModel(const std::string& name = "Vanaheimr");

public:
	MachineModel(const MachineModel&) = delete;
	const MachineModel& operator=(const MachineModel&) = delete;

public:
	/*! \brief Get the named physical register */
	const PhysicalRegister* getPhysicalRegister(RegisterId id) const;

public:
	/*! \brief Get the total register count */
	unsigned int totalRegisterCount() const;

public:
	const std::string name;

};

}

}


