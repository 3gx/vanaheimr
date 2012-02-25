/*! \file   PTXToVIRTranslator.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Sunday Fubruary 12, 2012
	\brief  The header file for the PTXToVIRTranslator class.
*/

#pragma once 

// Forward Declarations
                      namespace ir { class Module; }
                      namespace ir { class Kernel; }
                      namespace ir { class Global; }
namespace vanaheimr { namespace ir { class Module; } }


namespace vanaheimr
{

namespace translator
{

class PTXToVIRTranslator
{
public:
	typedef ::ir::Module PTXModule;
	typedef ::ir::Kernel PTXKernel;
	typedef ::ir::Global PTXGlobal;

	typedef vanaheimr::ir::Module VIRModule;

public:
	PTXToVIRTranslator(Compiler* compiler);
	
public:
	/*! \brief Translate the specified PTX module, adding it to the
		vanaheimr compiler */
	void translate(const PTXModule& m);

private:
	void _translateGlobal(const PTXGlobal&);
	void _translateKernel(const PTXKernel&);

private:
	Compiler*  _compiler;
	
	VIRModule*       _vir;
	const PTXModule* _ptx;

};

}

}

