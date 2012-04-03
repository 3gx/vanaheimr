/*	\file   Operand.h
	\date   Saturday January 22, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Operand class.
*/

#pragma once

namespace archaeopteryx
{

namespace ir
{

/*! \brief A primitive data type */
enum DataType
{
	i1,
	i8,
	i16,
	i32,
	i64,
	f32,
	f64,
	InvalidDataType
};

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
		InvalidOperand
	};
	
	/*! \brief A type to hold a register idenfitier */
	typedef unsigned int RegisterType;

public:
	/*! \brief The mode of the operand determines how it is accessed */
	OperandMode mode;	
};

typedef Operand::RegisterType RegisterType;

/*! \brief A register operand */
class RegisterOperand : public Operand
{
public:
	/*! \brief The register being accessed */
	RegisterType reg;
	/*! \brief The data type */
	DataType type;
};

/*! \brief An immediate operand */
class ImmediateOperand : public Operand
{
public:
	/*! \brief The immediate value */
	union
	{
		long long unsigned int uint;
		double                 fp;
	};

	/*! \brief The data type */
	DataType type;
};

/*! \brief A predicate operand */
class PredicateOperand : public Operand
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
	/*! \brief The register being accessed */
	RegisterType reg;
	/*! \brief The predicate modifier */
	PredicateModifier modifier;
};

/*! \brief An indirect operand */
class IndirectOperand : public Operand
{
public:
	/*! \brief The register being accessed */
	RegisterType reg;
	/*! \brief The offset to add to the register */
	int offset;
	/*! \brief The data type */
	DataType type;
};

/*! \brief A union over the different operand types */
union OperandContainer
{
	RegisterOperand  asRegister;
	ImmediateOperand asImmediate;
	PredicateOperand asPredicate;
	IndirectOperand  asIndirect;
	Operand          asOperand;
};

}

}

