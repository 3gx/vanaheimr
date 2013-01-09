/*! \file   ChaitanBriggsRegisterAllocatorPass.cpp
	\date   Wednesday January 2, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the ChaitanBriggsRegisterAllocatorPass class.
*/

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/ChaitanBriggsRegisterAllocatorPass.h>

#include <vanaheimr/machine/interface/MachineModel.h>

#include <vanaheimr/ir/interface/VirtualRegister.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

namespace vanaheimr
{

namespace codegen
{

ChaitanBriggsRegisterAllocatorPass::ChaitanBriggsRegisterAllocatorPass()
: RegisterAllocator({}, "ChaitanBriggsRegisterAllocatorPass")
{

}

void ChaitanBriggsRegisterAllocatorPass::runOnFunction(Function& f)
{
	assertM(false, "Not implemented.");
}

RegisterAllocator::VirtualRegisterSet
	ChaitanBriggsRegisterAllocatorPass::getSpilledRegisters()
{
	return _spilled;
}

const machine::PhysicalRegister*
	ChaitanBriggsRegisterAllocatorPass::getPhysicalRegister(
	const ir::VirtualRegister& vr) const
{
	auto allocatedRegister = _allocated.find(vr.id);
	
	if(allocatedRegister == _allocated.end()) return nullptr;
	
	return _machine->getPhysicalRegister(allocatedRegister->second);
}


}

}


