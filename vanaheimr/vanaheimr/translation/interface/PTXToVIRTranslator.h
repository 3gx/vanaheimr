/*! \file   PTXToVIRTranslator.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Sunday Fubruary 12, 2012
	\brief  The header file for the PTXToVIRTranslator class.
*/

#pragma once 

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Function.h>

// Standard Library Includes
#include <unordered_map>
#include <string>

// Forward Declarations
                      namespace ir { class Module;         }
                      namespace ir { class Kernel;         }
                      namespace ir { class Global;         }
                      namespace ir { class BasicBlock;     }
                      namespace ir { class PTXInstruction; }
                      namespace ir { class PTXOperand;     }
                      namespace ir { class PTXKernel;      }
namespace vanaheimr { namespace ir { class Module;         } }


namespace vanaheimr
{

namespace translation
{

class PTXToVIRTranslator
{
public:
	typedef ::ir::Module     PTXModule;

public:
	PTXToVIRTranslator(compiler::Compiler* compiler);
	
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
	typedef int                  PTXDataType;

	typedef unsigned int PTXRegisterId;

	typedef vanaheimr::ir::Module VIRModule;

private:
	void _translateGlobal(const PTXGlobal&);
	void _translateKernel(const PTXKernel&);
	void _translateRegisterValue(PTXRegisterId, PTXDataType);
	void _translateBasicBlock(const PTXBasicBlock&);

private:
	void _translateInstruction(const PTXInstruction& );

	bool _translateComplexInstruction(const PTXInstruction& );
	bool _translateSimpleUnaryInstruction(const PTXInstruction& );
	bool _translateSimpleBinaryInstruction(const PTXInstruction& );

private:
	typedef std::unordered_map<PTXRegisterId,
		ir::Function::register_iterator> RegisterMap;
	typedef std::unordered_map<std::string,
		ir::Function::iterator> BasicBlockMap;

private:
	ir::Operand* _newTranslatedOperand(const PTXOperand& ptx);
	ir::Operand* _newTranslatedPredicateOperand(const PTXOperand& ptx);

private:
	ir::VirtualRegister* _getRegister(PTXRegisterId id);
	ir::Variable*        _getGlobal(const std::string& name);
	ir::BasicBlock*      _getBasicBlock(const std::string& name);
	ir::Operand*         _getSpecialValueOperand(unsigned int id);
	ir::VirtualRegister* _newTemporaryRegister();
	const ir::Type*      _getType(PTXDataType type);

private:
	compiler::Compiler* _compiler;
	
	VIRModule*            _vir;
	const PTXModule*      _ptx;
	const PTXBasicBlock*  _block;
	const PTXInstruction* _instruction;
	
	RegisterMap   _registers;
	BasicBlockMap _blocks;
};

}

}

