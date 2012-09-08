/*! \file   Pass.cpp
	\author Gregory Diamos <gdiamos@nvidia.com>
	\date   Tuesday September 15, 2009
	\brief  The source file for the Pass class.
*/

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/PassManager.h>
#include <vanaheimr/transforms/interface/Pass.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

namespace vanaheimr
{

namespace transforms
{

Pass::Pass(Type t, const StringVector& a, const std::string& n)
	: type(t), analyses(a), name(n), _manager(0)
{

}

Pass::~Pass()
{

}

void Pass::setPassManager(PassManager* m)
{
	_manager = m;
}

Pass::Analysis* Pass::getAnalysis(const std::string& type)
{
	assert(_manager != 0);

	return _manager->getAnalysis(type);
}

const Pass::Analysis* Pass::getAnalysis(const std::string& type) const
{
	assert(_manager != 0);

	return _manager->getAnalysis(type);
}

Pass* Pass::getPass(const std::string& name)
{
	assert(_manager != 0);

	return _manager->getPass(name);
}

const Pass* Pass::getPass(const std::string& name) const
{
	assert(_manager != 0);

	return _manager->getPass(name);
}

void Pass::invalidateAnalysis(const std::string& type)
{
	assert(_manager != 0);

	return _manager->invalidateAnalysis(type);
}

Pass::StringVector Pass::getDependentPasses() const
{
	return StringVector();
}

void Pass::configure(const StringVector& options)
{

}

std::string Pass::toString() const
{
	return name;
}

ImmutablePass::ImmutablePass(const StringVector& a, const std::string& n) 
 : Pass(Pass::ImmutablePass, a, n)
{

}

ImmutablePass::~ImmutablePass()
{

}

ModulePass::ModulePass(const StringVector& a, const std::string& n) 
 : Pass( Pass::ModulePass, a, n)
{

}

ModulePass::~ModulePass()
{

}

FunctionPass::FunctionPass(const StringVector& a, const std::string& n)
 : Pass(Pass::FunctionPass, a, n)
{

}

FunctionPass::~FunctionPass()
{

}

ImmutableFunctionPass::ImmutableFunctionPass(
	const StringVector& a, const std::string& n)
 : Pass(Pass::ImmutableFunctionPass, a, n)
{

}

ImmutableFunctionPass::~ImmutableFunctionPass()
{

}

BasicBlockPass::BasicBlockPass(const StringVector& a, const std::string& n)
 : Pass(Pass::BasicBlockPass, a, n)
{

}

BasicBlockPass::~BasicBlockPass()
{

}

}

}
