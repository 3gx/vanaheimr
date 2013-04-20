/*! \file   Target.cpp
	\date   Thursday December 20, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Target class.
*/

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/Target.h>

#include <vanaheimr/codegen/interface/ArchaeopteryxTarget.h>

// Standard Library Includes
#include <map>
#include <cassert>

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

class TargetDatabase
{
public:
	typedef std::map<std::string, Target*> TargetMap;

public:
	TargetMap targets;

public:
	~TargetDatabase();
};

TargetDatabase::~TargetDatabase()
{
	for(auto target : targets)
	{
		delete target.second;
	}
}

static TargetDatabase targetDatabase;

void Target::registerTarget(Target* newTarget)
{
	assert(targetDatabase.targets.count(newTarget->name()) == 0);

	targetDatabase.targets.insert(std::make_pair(newTarget->name(), newTarget));
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


