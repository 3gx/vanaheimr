/*! \file   ChaitanBriggsRegisterAllocatorPass.cpp
	\date   Wednesday January 2, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the ChaitanBriggsRegisterAllocatorPass class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/RegisterAllocator.h>

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
	std::string getPhysicalRegisterName(const ir::VirtualRegister&);

private:
	typedef util::LargeMap<unsigned int, unsigned int> RegisterMap;

private:
	VirtualRegisterSet _spilled;
	RegisterMap        _allocated;
	
};

}

}


