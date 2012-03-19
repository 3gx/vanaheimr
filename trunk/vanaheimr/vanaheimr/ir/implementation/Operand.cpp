/*	\file   Operand.cpp
	\date   Saturday January 22, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the Operand class.
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Operand.h>

namespace vanaheimr
{

namespace ir
{

Operand::Operand(OperandMode mode, Instruction* instruction)
: _mode(mode), _instruction(instruction)
{

}

bool Operand::isRegister() const
{
	return mode() == Register || mode() == Immediate || mode() == Predicate;
}

Operand::OperandMode Operand::mode() const
{
	return _mode;
}

Instruction* Operand::instruction() const
{
	return _instruction;
}

RegisterOperand::RegisterOperand(VirtualRegister* reg, Instruction* i)
: Operand(Register, i), virtualRegister(reg)
{
	
}

Operand* RegisterOperand::clone() const
{
	return new RegisterOperand(*this);
}

RegisterOperand::RegisterOperand(VirtualRegister* reg, Instruction* i,
	OperandMode m)
: Operand(m, i), virtualRegister(reg)
{

}

ImmediateOperand::ImmediateOperand(uint64_t v, Instruction* i)
: Operand(Immediate, i)
{
	uint = v;
}

ImmediateOperand::ImmediateOperand(double d, Instruction* i)
: Operand(Immediate, i)
{
	fp = d;
}

Operand* ImmediateOperand::clone() const
{
	return new ImmediateOperand(*this);
}

PredicateOperand::PredicateOperand(VirtualRegister* reg,
	PredicateModifier mod, Instruction* i)
: RegisterOperand(reg, i, Predicate), modifier(mod)
{

}

Operand* PredicateOperand::clone() const
{
	return new PredicateOperand(*this);
}

IndirectOperand::IndirectOperand(VirtualRegister* reg, int64_t o,
	Instruction* i)
: RegisterOperand(reg, i, Indirect), offset(o)
{

}

Operand* IndirectOperand::clone() const
{
	return new IndirectOperand(*this);
}

AddressOperand::AddressOperand(Variable* value, Instruction* i)
: Operand(Address, i), globalValue(value)
{
	
}

Operand* AddressOperand::clone() const
{
	return new AddressOperand(*this);
}

}

}

