/*	\file   Instruction.h
	\date   Saturday January 22, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Instruction class.
*/

#pragma once

// Archaeopteryx Includes
#include <archaeopteryx/ir/interface/Operand.h>

/*! \brief A namespace for the compiler internal representation */
namespace ir
{

/*! \brief A space efficient class for representing a single instruction */
class Instruction
{
public:
	/*! \brief The set of possible instructions */
	enum Opcode
	{
		Add,
		And,
		Atom,
		Bar,
		Bitcast,
		Bra,
		Fma,
		Fpext,
		Fptosi,
		Fptoui,
		Fptrunc,
		Ld,
		Membar,
		Mov,
		Mul,
		Or,
		Ret,
		SelP,
		SetP,
		Sext,
		Sdiv,
		Shl,
		Shr,
		Sitofp,
		Srem,
		St,
		Sub,
		Trunc,
		Udiv,
		Uitofp,
		Urem,
		Xor,
		Zext,
		InvalidOpcode
	};
	
public:
	/*! \brief The instruction opcode */
	Opcode opcode;

	/*! \brief The guard predicate */
	PredicateOperand guard;

};

/*! \brief A unary instruction */
class UnaryInstruction : public Instruction
{
public:
	/*! \brief The destination operand. */
	Operand d;
	/*! \brief The source operand. */
	Operand a;

};

/*! \brief A binary instruction */
class BinaryInstruction : public Instruction
{
public:
	/*! \brief The destination operand. */
	Operand d;
	/*! \brief The first source operand. */
	Operand a;
	/*! \brief The second source operand. */
	Operand b;
};

/*! \brief An add instruction */
class Add : public BinaryInstruction
{
};

/*! \brief An and instruction */
class And : public BinaryInstruction
{
};

/*! \brief An atomic operation instruction */
class Atom : public BinaryInstruction
{
public:
	/*! \brief Possible atomic operations */
	enum Operation
	{
		AtomicAnd,
		AtomicOr,
		AtomicXor,
		AtomicCas,
		AtomicExch,
		AtomicAdd,
		AtomicInc,
		AtomicDec, 
		AtomicMin,
		AtomicMax,		
		InvalidOperation
	};

public:
	Operation operation;
	Operand   c;
};

/*! \brief Perform a thread group barrier */
class Bar : public Instruction
{
public:
	
};

/*! \brief Perform a raw bitcast */
class Bitcast : public UnaryInstruction
{

};

/*! \brief Perform a branch */
class Bra : public Instruction
{
public:
	enum BranchModifier
	{
		UniformBranch,
		MultitargetBranch,
		InvalidModifier
	};

public:
	Operand target;
	BranchModifier modifier;
};

/*! \brief A fused multiple add */
class Fma : public BinaryInstruction
{
public:
	Operand c;
};

/*! \brief A floating point precision extension instruction */
class Fpext : public UnaryInstruction
{

};

/*! \brief A floating point to signed integer instruction */
class Fptosi : public UnaryInstruction
{

};

/*! \brief A floating point to unsigned integer instruction */
class Fptoui : public UnaryInstruction
{

};

/*! \brief A floating point precision truncate instruction */
class Fptrunc : public UnaryInstruction
{

};

/*! \brief Load a value from memory */
class Ld : public UnaryInstruction
{

};

/*! \brief Wait until memory operations at the specified level have completed */
class Membar : public Instruction
{
public:
	enum Level
	{
		Thread,
		Cta,
		Kernel,
		InvalidLevel
	};

public:
	Level level;	
};

/*! \brief Move a value into a register */
class Mov : public UnaryInstruction
{

};


/*! \brief Multiply operands together */ 
		Mul,
		Or,
		Ret,
		SelP,
		SetP,
		Sext,
		Sdiv,
		Shl,
		Shr,
		Sitofp,
		Srem,
		St,
		Sub,
		Trunc,
		Udiv,
		Uitofp,
		Urem,
		Xor,
		Zext,

}

