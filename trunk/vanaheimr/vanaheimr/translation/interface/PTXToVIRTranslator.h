/*! \file   PTXToVIRTranslator.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Sunday Fubruary 12, 2012
	\brief  The header file for the PTXToVIRTranslator class.
*/

#pragma once 

// Forward Declarations
                      namespace ir { class Module;         }
                      namespace ir { class Kernel;         }
                      namespace ir { class Global;         }
                      namespace ir { class BasicBlock;     }
                      namespace ir { class PTXInstruction; }
namespace vanaheimr { namespace ir { class Module;         } }


namespace vanaheimr
{

namespace translator
{

class PTXToVIRTranslator
{
public:
	typedef ::ir::Module     PTXModule;

public:
	PTXToVIRTranslator(Compiler* compiler);
	
public:
	/*! \brief Translate the specified PTX module, adding it to the
		vanaheimr compiler */
	void translate(const PTXModule& m);

private:
	typedef ::ir::Kernel         PTXKernel;
	typedef ::ir::Global         PTXGlobal;
	typedef ::ir::BasicBlock     PTXBasicBlock;
	typedef ::ir::PTXInstruction PTXInstruction;
	typedef ::ir::PTXOperand     PTXOperand;

	typedef ::analysis::DataflowGraph::Register PTXRegister;

	typedef vanaheimr::ir::Module VIRModule;

	typedef std::unordered_map<PTXRegiserId,
		ir::Function::value_iterator> ValueMap;

private:
	void _translateGlobal(const PTXGlobal&);
	void _translateKernel(const PTXKernel&);
	void _translateBasicBlock(const PTXBasicBlock&);

	void _translateRegisterValue(const PTXRegister& );

private:
	void _translateInstruction(const PTXInstruction& );

	bool _translateComplexInstruction(const PTXInstruction& );
	bool _translateSimpleUnaryInstruction(const PTXInstruction& );
	bool _translateSimpleBinaryInstruction(const PTXInstruction& );

private:
	Compiler*  _compiler;
	
	VIRModule*       _vir;
	const PTXModule* _ptx;

	ValueMap _registerToValueMap;

};

}

}

