/*	\file   Instruction.h
	\date   Saturday January 22, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Instruction class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Operand.h>

// Standard Library Includes 
#include <vector>
#include <string>

namespace vanaheimr
{

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
		Ashr,
		Atom,
		Bar,
		Bitcast,
		Bra,
		Call,
		Fpext,
		Fptosi,
		Fptoui,
		Fptrunc,
		Launch,
		Ld,
		Lshr,
		Membar,
		Mul,
		Or,
		Ret,
		SetP,
		Sext,
		Sdiv,
		Shl,
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

	typedef std::vector<Operand*> OperandVector;

public:
	Instruction();

	Instruction(const Instruction&);
	Instruction& operator=(const Instruction&);

public:
	bool isLoad()   const;
	bool isStore()  const;
	bool isBranch() const;
	bool isCall()   const;

public:
	virtual Instruction* clone() const = 0;

public:
	static std::string toString(Opcode o);
	
public:
	/*! \brief The instruction opcode */
	Opcode opcode;

	/*! \brief The guard predicate */
	PredicateOperand guard;

	/*! \brief The set of all operands read by the instruction */
	OperandVector reads;
	/*! \brief The set of all operands written by the instruction */
	OperandVector writes;

};

/*! \brief A unary instruction */
class UnaryInstruction : public Instruction
{
public:
	UnaryInstruction();
	UnaryInstruction(const UnaryInstruction& i);

	~UnaryInstruction();

	UnaryInstruction& operator=(const UnaryInstruction& i);

public:
	/*! \brief The destination operand. */
	Operand* d;
	/*! \brief The source operand. */
	Operand* a;

};

/*! \brief A binary instruction */
class BinaryInstruction : public Instruction
{
public:
	BinaryInstruction();
	BinaryInstruction(const BinaryInstruction& i);

	~BinaryInstruction();

	BinaryInstruction& operator=(const BinaryInstruction& i);

public:
	Instruction* clone() const;

public:
	/*! \brief The destination operand. */
	Operand* d;
	/*! \brief The first source operand. */
	Operand* a;
	/*! \brief The second source operand. */
	Operand* b;
};

/*! \brief An instruction involving a comparison */
class ComparisonInstruction : public BinaryInstruction
{
public:
	/*! \brief All possible comparisons */
	enum Comparison
	{
		OrderedEqual,
		OrderedNotEqual,
		OrderedLessThan,
		OrderedLessOrEqual,
		OrderedGreater,
		OrderedGreaterOrEqual,
		UnorderedEqual,
		UnorderedNotEqual,
		UnorderedLessThan,
		UnorderedLessOrEqual,
		UnorderedGreaterThan,
		UnorderedGreaterOrEqual,
		IsANumber,
		NotANumber,
		InvalidComparison
	};
	
public:
	/*! \brief The comparison operation */
	Comparison comparison;
};

/*! \brief An add instruction */
class Add : public BinaryInstruction
{
};

/*! \brief An and instruction */
class And : public BinaryInstruction
{
};

/*! \brief Perform arithmetic shift right */
class Ashr : public BinaryInstruction
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
	Atom();
	Atom(const Atom& i);

	~Atom();

	Atom& operator=(const Atom& i);

public:
	Instruction* clone() const;

public:
	Operation operation;
	Operand*   c;
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
	Bra();
	Bra(const Bra& i);

	~Bra();

	Bra& operator=(const Bra& i);

public:
	virtual Instruction* clone() const;

public:
	Operand*       target;
	BranchModifier modifier;
};

/*! \brief Branch and save the return pc */
class Call : public Bra
{
public:
	Call();
	Call(const Call& i);

	~Call();

	Call& operator=(const Call& i);

public:
	Instruction* clone() const;

public:
	Operand* link;

};

/*! \brief Launch a new HTA at the specified entry point */
class Launch : public Instruction
{

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

/*! \brief Logical shift right */
class Lshr : public BinaryInstruction
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


/*! \brief Multiply two operands together */
class Mul : public BinaryInstruction
{

};

/*! \brief Perform a logical OR operation */
class Or : public BinaryInstruction
{

};

/*! \brief Return from the current function call, or exit */
class Ret : public UnaryInstruction
{

};

/*! \brief Compare two operands and set a third predicate */
class SetP : public ComparisonInstruction
{

};

/*! \brief Sign extend an integer */
class Sext : public UnaryInstruction
{

};

/*! \brief Perform signed division */
class Sdiv : public BinaryInstruction
{

};

/*! \brief Perform shift left */
class Shl : public BinaryInstruction
{

};

/*! \brief Convert a signed int to a floating point */
class Sitofp : public UnaryInstruction
{

};

/*! \brief Perform a signed remainder operation */
class Srem : public BinaryInstruction
{

};

/*! \brief Perform a store operation */
class St : public UnaryInstruction
{

};

/*! \brief Perform a subtract operation */
class Sub : public BinaryInstruction
{

};

/*! \brief Truncate an integer */
class Trunc : public UnaryInstruction
{
	
};

/*! \brief Perform an unsigned division operation */
class Udiv : public BinaryInstruction
{

};

/*! \brief Convert an unsigned int to a floating point */
class Uitofp : public UnaryInstruction
{

};

/*! \brief Perform an unsigned remainder operation */
class Urem : public BinaryInstruction
{

};

/*! \brief Perform a logical OR operation */
class Xor : public BinaryInstruction
{

};

/*! \brief Zero extend an integer */
class Zext : public UnaryInstruction
{
	
};

}

}

