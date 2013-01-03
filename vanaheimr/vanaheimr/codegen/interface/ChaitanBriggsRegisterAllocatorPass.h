/*! \file   ChaitanBriggsRegisterAllocatorPass.h
	\date   Wednesday January 2, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the ChaitanBriggsRegisterAllocatorPass class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/RegisterAllocator.h>

#include <vanaheimr/util/interface/LargeMap.h>

// Forward Declarations
namespace vanaheimr { namespace machine { class MachineModel; } }

namespace vanaheimr
{

namespace codegen
{

class ChaitanBriggsRegisterAllocatorPass : public RegisterAllocator
{
public:
	ChaitanBriggsRegisterAllocatorPass();

public:
	/*! \brief Run the pass on a specific function in the module */
	void runOnFunction(Function& f);

public:
	/*! \brief Get the set of values that were spilled during allocation */
	VirtualRegisterSet getSpilledRegisters();
	
	/*! \brief Get the mapping of a value to a named physical register */
	const machine::PhysicalRegister* getPhysicalRegisterName(
		const ir::VirtualRegister&);

private:
	typedef util::LargeMap<unsigned int, unsigned int> RegisterMap;

private:
	VirtualRegisterSet _spilled;
	RegisterMap        _allocated;

private:
	const machine::MachineModel* _machine;
};

}

}


