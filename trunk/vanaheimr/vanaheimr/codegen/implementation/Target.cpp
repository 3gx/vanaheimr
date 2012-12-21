/*! \file   Target.cpp
	\date   Thursday December 20, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Target class.
*/

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/Target.h>

#include <vanaheimr/codegen/interface/ArchaeopteryxTarget.h>

namespace vanaheimr
{

namespace codegen
{

Target::Target(const std::string& name)
: _module(nullptr), _name(name)
{

}

Target::~Target()
{

}

Target* Target::createTarget(const std::string& name)
{
	if(name == "ArchaeopteryxSimulatorTarget")
	{
		return new ArchaeopteryxTarget;
	}
	
	return nullptr;
}

void Target::assignModule(ir::Module* module)
{
	_module = module;
}

const std::string& Target::name() const
{
	return _name;
}

}

}


