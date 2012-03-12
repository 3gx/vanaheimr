/*! \file   PTXToVIRTranslator.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Sunday Fubruary 12, 2012
	\brief  The header file for the PTXToVIRTranslator class.
*/

#pragma once 

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/Variable.h>

// Standard Library Includes
#include <unordered_map>
#include <string>

// Forward Declarations
                      namespace ir       { class Module;           }
                      namespace ir       { class PTXKernel;        }
                      namespace ir       { class Global;           }
                      namespace ir       { class BasicBlock;       }
                      namespace ir       { class PTXInstruction;   }
                      namespace ir       { class PTXOperand;       }
                      namespace ir       { class PTXKernel;        }
namespace vanaheimr { namespace ir       { class Module;           } }
namespace vanaheimr { namespace ir       { class PredicateOperand; } }
namespace vanaheimr { namespace ir       { class Constant;         } }
namespace vanaheimr { namespace compiler { class Compiler;         } }

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
	typedef ::ir::PTXKernel      PTXKernel;
	typedef ::ir::Global         PTXGlobal;
	typedef ::ir::BasicBlock     PTXBasicBlock;
	typedef ::ir::PTXInstruction PTXInstruction;
	typedef ::ir::PTXOperand     PTXOperand;
	typedef int                  PTXDataType;

	typedef unsigned int PTXRegisterId;
	typedef unsigned int PTXAttribute;
	typedef unsigned int PTXLinkingDirective;

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
	ir::Operand*         _newTranslatedOperand(const PTXOperand& ptx);
	ir::PredicateOperand _translatePredicateOperand(const PTXOperand& ptx);

private:
	ir::VirtualRegister*  _getRegister(PTXRegisterId id);
	ir::Variable*         _getGlobal(const std::string& name);
	ir::Variable*         _getBasicBlock(const std::string& name);
	ir::Operand*          _getSpecialValueOperand(unsigned int id);
	ir::VirtualRegister*  _newTemporaryRegister();
	const ir::Type*       _getType(PTXDataType type);
	const ir::Type*       _getType(const std::string& name);
	ir::Variable::Linkage _translateLinkage(PTXAttribute linkage);
	ir::Variable::Linkage _translateLinkingDirective(PTXLinkingDirective d);
	ir::Constant*         _translateInitializer(const PTXGlobal& g);
	
private:
	compiler::Compiler* _compiler;
	ir::Module*         _module;
	ir::Function*       _function;
	ir::BasicBlock*     _block;
	ir::Instruction*    _instruction;
	
	const PTXModule*      _ptx;
	
	RegisterMap   _registers;
	BasicBlockMap _blocks;
	
};

}

}

