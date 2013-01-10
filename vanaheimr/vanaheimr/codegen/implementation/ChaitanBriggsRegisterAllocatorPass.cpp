/*! \file   ChaitinBriggsRegisterAllocatorPass.cpp
	\date   Wednesday January 2, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the ChaitinBriggsRegisterAllocatorPass class.
*/

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/ChaitinBriggsRegisterAllocatorPass.h>

#include <vanaheimr/machine/interface/MachineModel.h>

#include <vanaheimr/ir/interface/VirtualRegister.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

namespace vanaheimr
{

namespace codegen
{

ChaitinBriggsRegisterAllocatorPass::ChaitinBriggsRegisterAllocatorPass()
: RegisterAllocator({}, "ChaitinBriggsRegisterAllocatorPass")
{

}

typedef util::SmallSet<ir::VirtualRegister*> VirtualRegisterSet;
typedef util::LargeMap<ir::VirtualRegister*, VirtualRegisterSet>
	InterferenceMap;

static InterferenceMap buildInterferences(Function& f);

void ChaitinBriggsRegisterAllocatorPass::runOnFunction(Function& f)
{
	auto interferences = buildInterferences(f);
	
	// attempt to color the interferences
	color(_spilled, _allocated, f, interferences);
}

RegisterAllocator::VirtualRegisterSet
	ChaitinBriggsRegisterAllocatorPass::getSpilledRegisters()
{
	return _spilled;
}

const machine::PhysicalRegister*
	ChaitinBriggsRegisterAllocatorPass::getPhysicalRegister(
	const ir::VirtualRegister& vr) const
{
	auto allocatedRegister = _allocated.find(vr.id);
	
	if(allocatedRegister == _allocated.end()) return nullptr;
	
	return _machine->getPhysicalRegister(allocatedRegister->second);
}


}

}


