/*! \file   MachineModel.h
	\date   Thursday January 3, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the MachineModel class.
*/

// Vanaheimr Includes
#include <vanaheimr/machine/interface/MachineModel.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

namespace vanaheimr
{

namespace machine
{

MachineModel::MachineModel(const std::string& n)
: name(n)
{

}

const PhysicalRegister* MachineModel::getPhysicalRegister(RegisterId id) const
{
	assertM(false, "Not implemented.");
	
	return nullptr;
}

unsigned int MachineModel::totalRegisterCount() const
{
	assertM(false, "Not implemented.");

	return 0;
}

void MachineModel::configure(const StringVector& )
{
	// TODO
}

}

}


