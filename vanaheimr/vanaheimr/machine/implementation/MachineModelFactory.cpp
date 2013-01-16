/*! \file   MachineModelFactory.h
	\date   Wednesday January 16, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the MachineModelFactory class.
*/

// Vanaheimr Includes
#include <vanaheimr/machine/interface/MachineModelFactory.h>

#include <vanaheimr/machine/interface/MachineModel.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

namespace vanaheimr
{

namespace machine
{

MachineModel* MachineModelFactory::createMachineModel(const std::string& name,
	const StringVector& options)
{
	MachineModel* machine = nullptr;

	if(machine != nullptr)
	{
		machine->configure(options);
	}

	return machine;
}

MachineModel* MachineModelFactory::createDefaultMachine()
{
	return createMachineModel("ArchaeopteryxSimulator");
}

}

}

