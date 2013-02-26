/*! \file   Instruction.h
	\date   Wednesday February 1, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the Operation class.

	The best documentation is clear, understandable code.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Instruction.h>

// Standard Library Includes
#include <vector>
#include <list>

// Forward Declaration
namespace vanaheimr { namespace machine { class Operation; } }

namespace vanaheimr
{

namespace machine
{

/*! \brief A model for an abstract machine instruction */
class Instruction : public vanaheimr::ir::Instruction
{
public:
	Instruction(const Operation* op, BasicBlock* block = 0, Id id = 0);
	~Instruction();
	
public:
	Instruction& operator=(const Instruction&);
	Instruction(const Instruction&);
	
public:
	virtual std::string toString() const;
	virtual std::string modifierString() const;

public:
	virtual Instruction* clone() const;

public:
	 /*! \brief The machine operation performed by the instruction. */
	const Operation* operation;

};

}

}


