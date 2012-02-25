/*! \file   PTXToVIRTranslator.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Saturday Fubruary 25, 2012
	\brief  The source file for the PTXToVIRTranslator class.
*/

// Vanaheimr Includes
#include <vanaheimr/translation/interface/PTXToVIRTranslator.h>

namespace vanaheimr
{

namespace translator
{

PTXToVIRTranslator::PTXToVIRTranslator(Compiler* compiler)
: _compiler(compiler)
{

}

void PTXToVIRTranslator::translate(const PTXModule& m)
{
	_ptx = &m;
	_vir = &*_compiler->newModule(m.name);
	
	// Translate globals
	for(PTXModule::GlobalMap::const_iterator global = m.globals().begin();
		global != m.globals().end(); ++global)
	{
		_translateGlobal(global->second);
	}
	
	// Translate kernel functions
	for(PTXModule::KernelMap::const_iterator kernel = m.kernels().begin();
		kernel != m.kernels.().end(); ++kernel)
	{
		_translateFunction(kernel->second);
	}
}

void PTXToVIRTranslator::_translateGlobal(const PTXGlobal& global)
{
	VIRModule::global_iterator virGlobal = _vir->newGlobal(global.name,
		_translateType(global.statement.type),
		_translateLinkage(global.statement.attribute));
		
	if(global.initializedBytes() != 0)
	{
		virGlobal->setInitializer(_translateInitializer(global));
	}
}

void PTXToVIRTranslator::_translateKernel(const PTXKernel& kernel);

}

}

