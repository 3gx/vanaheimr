/*	\file   Operand.h
	\date   Saturday January 22, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Operand class.
*/

#pragma once

// Standard Library Includes
#include <cstdint>
#include <string>

// Forward Declarations
namespace vanaheimr { namespace ir { class Argument;        } }
namespace vanaheimr { namespace ir { class VirtualRegister; } }
namespace vanaheimr { namespace ir { class Variable;        } }
namespace vanaheimr { namespace ir { class Instruction;     } }
namespace vanaheimr { namespace ir { class Type;            } }

namespace vanaheimr
{

namespace ir
{

/*! \brief Represents a single instruction operand */
class Operand
{
public:
	/*! \brief An operand type */
	enum OperandMode
	{
		Register,
		Immediate,
		Predicate,
		Indirect,
		Address,
		Argument
	};
	
	/*! \brief A type to hold a register idenfitier */
	typedef unsigned int RegisterType;

public:
	Operand(OperandMode mode, Instruction* instruction);

public:
	/*! \brief Is the operand a register */
	bool isRegister() const;

public:
	/*! \brief The mode of the operand determines how it is accessed */
	OperandMode mode() const;

	/*! \brief The owning instruction */
	Instruction* instruction() const;

public:
	virtual Operand* clone() const = 0;
	virtual std::string toString() const = 0;

private:
	OperandMode  _mode;
	Instruction* _instruction;
};

typedef Operand::RegisterType RegisterType;

/*! \brief A register operand */
class RegisterOperand : public Operand
{
public:
	RegisterOperand(VirtualRegister* reg, Instruction* i);

public:
	virtual Operand* clone() const;
	virtual std::string toString() const;


public:
	/*! \brief The register being accessed */
	VirtualRegister* virtualRegister;

protected:
	RegisterOperand(VirtualRegister* reg, Instruction* i, OperandMode m);

};

/*! \brief An immediate operand */
class ImmediateOperand : public Operand
{
public:
	ImmediateOperand(uint64_t v, Instruction* i);
	ImmediateOperand(double   d, Instruction* i);

public:
	Operand* clone() const;
	std::string toString() const;

public:
	/*! \brief The immediate value */
	union
	{
		uint64_t uint;
		double   fp;
	};

	/*! \brief The data type */
	const Type* type;
};

/*! \brief A predicate operand */
class PredicateOperand : public RegisterOperand
{
public:
	/*! \brief The modifier on the predicate */
	enum PredicateModifier
	{
		StraightPredicate,
		InversePredicate,
		PredicateTrue,
		PredicateFalse
	};

public:
	PredicateOperand(VirtualRegister* reg,
		PredicateModifier mod, Instruction* i);

public:
	bool isAlwaysTrue() const;

public:
	Operand* clone() const;
	std::string toString() const;

public:
	/*! \brief The predicate modifier */
	PredicateModifier modifier;
};

/*! \brief An indirect operand */
class IndirectOperand : public RegisterOperand
{
public:
	IndirectOperand(VirtualRegister* reg, int64_t offset, Instruction* i);

public:
	Operand* clone() const;
	std::string toString() const;

public:
	/*! \brief The offset to add to the register */
	int64_t offset;
};

/*! \brief An address operand */
class AddressOperand : public Operand
{
public:
	AddressOperand(Variable* value, Instruction* i);

public:
	Operand* clone() const;
	std::string toString() const;

public:
	Variable* globalValue;
};

class ArgumentOperand : public Operand
{
public:
	ArgumentOperand(ir::Argument* a, Instruction* i);

public:
	Operand* clone() const;
	std::string toString() const;
	
public:
	ir::Argument* argument;

};

}

}
