/*	\file   Operand.h
	\date   Saturday January 22, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Operand class.
*/

#pragma once

// Forwar Declarations
namespace vanaheimr { namespace ir { class VirtualRegister; } }

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
		InvalidOperand
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

private:
	OperandMode  _mode;
	Instruction* _instruction;
};

typedef Operand::RegisterType RegisterType;

/*! \brief A register operand */
class RegisterOperand : public Operand
{
public:
	RegisterOperand(Value* reg, Instruction* i);

public:
	/*! \brief The register being accessed */
	Value* reg;
};

/*! \brief An immediate operand */
class ImmediateOperand : public Operand
{
public:
	ImmediateOperand(uint64_t v, Instruction* i);
	ImmediateOperand(double   d, Instruction* i);

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
		PredicateFalse,
		InvalidPredicate
	};

public:
	PredicateOperand(Value* reg, PredicateModifier mod, Instruction* i);

public:
	/*! \brief The predicate modifier */
	PredicateModifier modifier;
};

/*! \brief An indirect operand */
class IndirectOperand : public RegisterOperand
{
public:
	IndirectOperand(Value* reg, int64_t offset, Instruction* i);

public:
	/*! \brief The offset to add to the register */
	int64_t offset;
};

/*! \brief An address operand */
class AddressOperand : public Operand
{
public:
	AddressOperand(GlobalValue* value, Instruction* i);

public:
	GlobalValue* globalValue;
};

}

}

