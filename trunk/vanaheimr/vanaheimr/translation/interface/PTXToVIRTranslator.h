/*! \file   PTXToVIRTranslator.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Sunday Fubruary 12, 2012
	\brief  The header file for the PTXToVIRTranslator class.
*/

#pragma once 

// Forward Declarations
namespace ir { class Module; }


namespace vanaheimr
{

namespace translator
{

class PTXToVIRTranslator
{
public:
	typedef ::ir::Module PTXModule;

public:
	PTXToVIRTranslator(Compiler* compiler);
	
public:
	/*! \brief Translate the specified PTX module, adding it to the
		vanaheimr compiler */
	void translate(const PTXModule& m);

private:
	Compiler* _compiler;

};

}

}

