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
		_translateKernel(kernel->second);
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

void PTXToVIRTranslator::_translateKernel(const PTXKernel& kernel)
{
	VIRModule::iterator function = _vir->newFunction(kernel.name(),
		Variable::ExternalLinkage);
	
	_function = &*function;
	
	// Translate Values
	PTXKernel::RegisterVector registers = kernel.getReferencedRegsisters();
	
	for(PTXKernel::RegisterVector::iterator reg = registers.begin();
		reg != registers.end(); ++reg)
	{
		_translateRegisterValue(**reg);
	}
	
	::ir::ControlFlowGraph::ConstBlockPointerVector sequence =
		kernel.cfg()->executable_sequence();
	
	for(::ir::ControlFlowGraph::ConstBlockPointerVector::iterator
		block = sequence.begin(); block != sequence.end(); ++block)
	{
		_translateBasicBlock(**block);
	}
}

void PTXToVIRTranslator::_translateRegisterValue(const PTXRegister& reg)
{
	std::stringstream name;
	
	name << "r"  << reg.reg;
	
	if(_registers.count(reg.reg) != 0)
	{
		throw std::runtime_error("Added duplicate virtual register '"
			+ name.str() + "'");
	}
	
	ir::Function::register_iterator newRegister = _function->newVirtualRegister(
		_getType(reg.type), name.str());

	_registers.insert(std::make_pair(reg.reg, newRegister));
}

void PTXToVIRTranslator::_translateBasicBlock(const PTXBasicBlock& basicBlock)
{
	VIRFunction::iterator block = _function->newBlock(_function->exit_block(),
		basicBlock.name);
		
	_block = &*block;
	
	for(PTXBasicBlock::const_instruction_iterator
		instruction = basicBlock.instructions.begin();
		instruction != basicBlock.instructions.end(); ++instruction)
	{
		const PTXInstruction& ptx = static_cast<const PTXInstruction&>(
			**instruction);
	
		_translateInstruction(ptx);
	}
}

void PTXToVIRTranslator::_translateInstruction(const PTXInstruction& ptx)
{
	// Translate complex instructions
	if(_translateComplexInstruction(ptx)) return;
	
	// Translate simple instructions
	if(_translateSimpleBinaryInstruction(ptx)) return;
	if(_translateSimpleUnaryInstruction(ptx))  return;
	
	assertM(false, "No translation implemented for instruction "
		<< ptx.toString());
}

bool PTXToVIRTranslator::_translateComplexInstruction(const PTXInstruction& ptx)
{
	return false;
}

static ir::UnaryInstruction* newUnaryInstruction(const PTXInstruction& ptx)
{
	switch(ptx.opcode)
	{
	case PTXInstruction::Ld: // fall through
	case PTXInstruction::Ldu:
	{
		return new ir::Ld;
	}
	case PTXInstruction::Mov:
	{
		return new ir::Bitcast;
	}
	case PTXInstruction::St:
	{
		return new ir::St;
	}
	case PTXInstruction::Cvt:
	{
		if(PTXOperand::isFloat(ptx.d))
		{
			if(PTXOperand::isFloat(ptx.a))
			{
				if(ptx.d.type == PTXOperand::f32)
				{
					if(ptx.a.type == PTXOperand::f32)
					{
						return new ir::Bitcast;
					}
					else
					{
						return new ir::Fptrunc;
					}
				}
				else
				{
					if(ptx.a.type == PTXOperand::f32)
					{
						return new ir::Fpext;
					}
					else
					{
						return new ir::Bitcast;
					}
				}
			}
			else if(PTXOperand::isSigned(ptx.a))
			{
				return new ir::Sitofp;
			}
			else
			{
				return new ir::Uitofp;
			}
		}
		else if(PTXOperand::isSigned(ptx.d))
		{
			if(PTXOperand::isFloat(ptx.a))
			{
				return new ir::Fptosi;
			}
			else
			{
				if(PTXOperand::bytes(ptx.a) > PTXOperand::bytes(ptx.d))
				{
					return new ir::Trunc;
				}
				else if(PTXOperand::bytes(ptx.d == PTXOperand::bytes(ptx.a))
				{
					return new ir::Bitcast;
				}
				else if(PTXOperand::isSigned(ptx.a))
				{
					return new ir::Sext;
				}
				else
				{
					return new ir::Zext;
				}
			}
		}
		else
		{
			if(PTXOperand::isFloat(ptx.a))
			{
				return new ir::Fptoui;
			}
			else
			{
				if(PTXOperand::bytes(ptx.a) > PTXOperand::bytes(ptx.d))
				{
					return new ir::Trunc;
				}
				else if(PTXOperand::bytes(ptx.d == PTXOperand::bytes(ptx.a))
				{
					return new ir::Bitcast;
				}
				else
				{
					return new ir::Zext;
				}
			}
		}
		break;
	}
	default:
	{
		break;
	}
	}
	
	return 0;	
}

ir::Operand* PTXToVIRTranslator::_newTranslatedPredicateOperand(
	const PTXOperand& ptx)
{
	if(ptx.addressMode != PTXOperand::Register)
	{
		throw std::runtime_error("Predicate operands must be registers.");
	}
	
	return new ir::PredicateOperand(_getRegister(ptx.reg), _instruction);
}

static bool isSimpleUnaryInstruction(const PTXInstruction& ptx)
{
	switch(ptx.opcode)
	{
	case PTXInstruction::Ld:
	case PTXInstruction::Ldu:
	case PTXInstruction::Mov:
	case PTXInstruction::St:
	{
		return true;
		break;
	}
	case PTXInstruction::Cvt:
	{
		if(ptx.modifier == PTXInstruction::Modifier_invalid)
		{
			return true;
		}
		break;
	}
	default:
	{
		break;
	}
	}
	
	return false;
}

bool PTXToVIRTranslator::_translateSimpleUnaryInstruction(
	const PTXInstruction& ptx)
{
	if(!isSimpleUnaryInstruction(ptx)) return false;
	
	ir::UnaryInstruction* vir = newUnaryInstruction(ptx);
	
	vir->guard = _newTranslatedPredicateOperand(ptx.pg);
	vir->d     = _newTranslatedOperand(ptx.d);
	vir->a     = _newTranslatedOperand(ptx.a);
	
	_block->push_back(vir);
	
	return true;
}

static bool isSimpleBinaryInstruction(const PTXInstruction& ptx)
{
	switch(ptx.opcode)
	{
	case PTXInstruction::Add: // fall through
	case PTXInstruction::And: // fall through
	case PTXInstruction::Div: // fall through
	case PTXInstruction::Mul: // fall through
	case PTXInstruction::Not: // fall through
	case PTXInstruction::Or:  // fall through
	case PTXInstruction::Rem: // fall through
	case PTXInstruction::Shl: // fall through
	case PTXInstruction::Sub: // fall through
	case PTXInstruction::Xor:
	{
		return true;
	}
	default:
	{
		break;
	}
	}
	
	return false;
}

static ir::BinaryInstruction* newBinaryInstruction(const PTXInstruction& ptx)
{
	switch(ptx.opcode)
	{
	case PTXInstruction::Add:
	{
		return new ir::Add;
	}
	case PTXInstruction::And:
	{
		return new ir::And;		
	}
	case PTXInstruction::Div:
	{
		return new ir::Div;		
	}
	case PTXInstruction::Mul:
	{
		return new ir::Mul;		
	}
	case PTXInstruction::Not:
	{
		return new ir::Not;		
	}
	case PTXInstruction::Or:
	{
		return new ir::Or;		
	}
	case PTXInstruction::Rem:
	{
		return new ir::Rem;		
	}
	case PTXInstruction::Shl:
	{
		return new ir::Shl;		
	}
	case PTXInstruction::Sub:
	{
		return new ir::Sub;		
	}
	case PTXInstruction::Xor:
	{
		return new ir::Xor;		
	}
	default:
	{
		break;
	}
	}
	
	return 0;
}

bool PTXToVIRTranslator::_translateSimpleBinaryInstruction(
	const PTXInstruction& ptx)
{
	if(!isSimpleBinaryInstruction(ptx)) return false;
	
	ir::BinaryInstruction* vir = newBinaryInstruction(ptx);
	
	vir->guard = _newTranslatedPredicatedOperand(ptx.pg);
	vir->d     = _newTranslatedOperand(ptx.d);
	vir->a     = _newTranslatedOperand(ptx.a);
	vir->b     = _newTranslatedOperand(ptx.d);
	
	_block->push_back(vir);
	
	return true;
}

ir::Operand* PTXToVIRTranslator::_newTranslatedOperand(const PTXOperand& ptx)
{
	switch(ptx.addressMode)
	{
	case PTXOperand::Register:
	{
		return new ir::RegisterOperand(_getRegister(ptx.reg), _instruction);
	}
	case PTXOperand::Indirect:
	{
		return new ir::IndirectOperand(_getRegister(ptx.reg),
			ptx.offset, _instruction);
	}
	case PTXOperand::Immediate:
	{
		return new ir::ImmediateOperand(ptx.imm_uint, _instruction);
	}
	case PTXOperand::Address:
	{
		return new ir::AddressOperand(_getGlobal(ptx.name), _instruction);
	}
	case PTXOperand::Label:
	{
		return new ir::AddressOperand(_getBasicBlock(ptx.name),
			_instruction);
	}
	case PTXOperand::Special:
	{
		return _getSpecialValueOperand(ptx.special);
	}
	case PTXOperand::BitBucket:
	{
		return new ir::RegisterOperand(_newTemporaryRegister(), _instruction);
	}
	default: break;
	}
	
	throw std::runtime_error("No translation implemented for "
		+ ptx.toString());
}

ir::Operand* PTXToVIRTranslator::_newTranslatedPredicateOperand(
	const PTXOperand& ptx)
{
	VirtualRegister* predicateRegister = 0;

	if(ptx.condition != PTXOperand::PT && ptx.condition != PTXOperand::nPT)
	{
		predicateRegister = _getRegister(ptx.reg);
	}
	
	return new ir::PredicateOperand(_getRegister(ptx.reg),
		_translatePredicateCondition(ptx.condition), _instruction);
}

ir::VirtualRegister* PTXToVIRTranslator::_getRegister(PTXRegisterId id)
{
	RegisterMap::iterator reg = _registers.find(id);
	
	if(reg == _registers.end())
	{
		std::stringstream name;
		
		name << "r" << id;

		throw std::runtime_error("PTX register " + name.str()
			+ " used without declaration.");
	}
	
	return &*reg->second;
}

ir::Variable* PTXToVIRTranslator::_getGlobal(const std::string& name);

ir::Variable* PTXToVIRTranslator::_getBasicBlock(const std::string& name);

ir::Operand* PTXToVIRTranslator::_getSpecialValueOperand(unsigned int id)
{

}

ir::VirtualRegiser* PTXToVIRTranslator::_newTemporaryRegister()
{
	ir::Function::register_iterator temp = _function->newVirtualRegister(
		_getType("i64"));
		
	return &*temp;
}

static std::string translateTypeName(::ir::PTXOperand::DataType type)
{
	switch(type)
	{
	case ::ir::PTXOperand::b8:  /* fall through */
	case ::ir::PTXOperand::s8:  /* fall through */
	case ::ir::PTXOperand::u8:
	{
		return "i8";
	}
	case ::ir::PTXOperand::s16: /* fall through */
	case ::ir::PTXOperand::u16: /* fall through */
	case ::ir::PTXOperand::b16:
	{
		return "i16";
	}
	case ::ir::PTXOperand::s32: /* fall through */
	case ::ir::PTXOperand::b32: /* fall through */
	case ::ir::PTXOperand::u32:
	{
		return "i32";
	}
	case ::ir::PTXOperand::s64: /* fall through */
	case ::ir::PTXOperand::b64: /* fall through */
	case ::ir::PTXOperand::u64:
	{
		return "i64";
	}
	case ::ir::PTXOperand::f32:
	{
		return "f32";
	}
	case ::ir::PTXOperand::f64:
	{
		return "f64";
	}
	case ::ir::PTXOperand::pred:
	{
		return "i1";
	}
	default: break;
	}
	
	return "";
}

const ir::Type* PTXToVIRTranslator::_getType(PTXDataType type)
{
	const ir::Type* type = _compiler->getType(translateTypeName(type));

	if(type == 0)
	{
		throw std::runtime_error("PTX translated type name '"
			+ translateTypeName(type) + "' is not a valid Vanaheimr type.");
	}
	
	return type;
}

}

}

