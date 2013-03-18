/*! \file   Instruction.h
	\date   Wednesday February 1, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the Instruction class.
*/


// Vanaheimr Includes
#include <vanaheimr/machine/interface/Instruction.h>

#include <vanaheimr/machine/interface/Operation.h>

namespace vanaheimr
{

namespace machine
{


Instruction::Instruction(const Operation* op, BasicBlock* block, Id id)
 : vanaheimr::ir::Instruction(Machine, block, id), operation(op)
{

}

std::string Instruction::opcodeString() const
{
	return operation->name;
}

vanaheimr::ir::Instruction* Instruction::clone() const
{
	return new Instruction(*this);
}

}

}


