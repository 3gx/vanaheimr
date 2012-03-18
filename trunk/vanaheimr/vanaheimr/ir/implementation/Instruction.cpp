/*	\file   Instruction.h
	\date   Saturday January 22, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Instruction class.
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Instruction.h>

namespace vanaheimr
{

/*! \brief A namespace for the compiler internal representation */
namespace ir
{

Instruction::Instruction(Opcode o, BasicBlock* b)
: opcode(o), block(b)
{

}

Instruction::~Instruction
{
	clear();
}

Instruction::Instruction(const Instruction& i)
: opcode(i.opcode), block(i.block)
{
	for(auto operand : i.read)
	{
		if(operand != 0)
		{
			reads.push_back(operand->clone());
		}
		else
		{
			reads.push_back(0);
		}
	}
	
	guard = reads[0];
	
	for(auto operand : i.write)
	{
		if(operand != 0)
		{
			writes.push_back(operand->clone());
		}
		else
		{
			writes.push_back(0);
		}
	}
}

Instruction& Instruction::operator=(const Instruction& i)
{
	if(this == &i) return *this;
	
	clear();
	
	opcode = i.opcode;
	block  = i.block;
	
	for(auto operand : i.read)
	{
		if(operand != 0)
		{
			reads.push_back(operand->clone());
		}
		else
		{
			reads.push_back(0);
		}
	}
	
	guard = reads[0];
	
	for(auto operand : i.write)
	{
		if(operand != 0)
		{
			writes.push_back(operand->clone());
		}
		else
		{
			writes.push_back(0);
		}
	}
	
	return *this;
}

bool Instruction::isLoad() const
{
	return opcode == Ld || opcode == Atom;
}

bool Instruction::isStore() const
{
	return opcode == St || opcode == Atom;
}

bool Instruction::isBranch() const
{
	return opcode == Bra || opcode == Call ;
}

bool Instruction::isCall() const
{
	return opcode == Call;
}

void Instruction::clear()
{
	for(auto operand : reads)  delete operand;
	for(auto operand : writes) delete operand;
	
	reads.clear();
	writes.clear();
}

static std::string Instruction::toString(Opcode o)
{
	switch(o)
	{
	case Add:     return "Add";
	case And:     return "And";
	case Ashr:    return "Ashr";
	case Atom:    return "Atom";
	case Bar:     return "Bar";
	case Bitcast: return "Bitcast";
	case Bra:     return "Bra";
	case Call:    return "Call";
	case Fdiv:    return "Fdiv";
	case Fmul:    return "Fmul";
	case Fpext:   return "Fpext";
	case Fptosi:  return "Fptosi";
	case Fptoui:  return "Fptoui";
	case Fptrunc: return "Fptrunc";
	case Frem:    return "Frem";
	case Launch:  return "Launch";
	case Ld:      return "Ld";
	case Lshr:    return "Lshr";
	case Membar:  return "Membar";
	case Mul:     return "Mul";
	case Or:      return "Or";
	case Ret:     return "Ret";
	case Setp:    return "Setp";
	case Sext:    return "Sext";
	case Sdiv:    return "Sdiv";
	case Shl:     return "Shl";
	case Sitofp:  return "Sitofp";
	case Srem:    return "Srem";
	case St:      return "St";
	case Sub:     return "Sub";
	case Trunc:   return "Trunc";
	case Udiv:    return "Udiv";
	case Uitofp:  return "Uitofp";
	case Urem:    return "Urem";
	case Xor:     return "Xor";
	case Zext:    return "Zext";
	default:      break;
	}
	
	return "InvalidOpcode";
}

UnaryInstruction::UnaryInstruction(Opcode o, BasicBlock* b)
: Instruction(o, b), d(0), a(0)
{
	writes.push_back(d);
	 reads.push_back(a);
}

UnaryInstruction::UnaryInstruction(const UnaryInstruction& i)
: Instruction(*i), d(writes.back()), a(reads.back())
{
	
}

UnaryInstruction& UnaryInstruction::operator=(const UnaryInstruction& i)
{
	if(&i == this) return *this;
	
	Instruction::operator=(i);
	
	d = writes[0];
	a =  reads[1];
	
	return *this;
}

BinaryInstruction::BinaryInstruction()
: Instruction(o, b), d(0), a(0), b(0)
{
	writes.push_back(d);
	 reads.push_back(a);
	 reads.push_back(b);
}

BinaryInstruction::BinaryInstruction(const BinaryInstruction& i)
: Instruction(*i), d(writes.back()), a(reads[1]), b(reads[2])
{
	
}

BinaryInstruction& BinaryInstruction::operator=(const BinaryInstruction& i)
{
	if(&i == this) return *this;
	
	Instruction::operator=(i);
	
	d = writes[0];
	a =  reads[1];
	b =  reads[2];
	
	return *this;
}

ComparisonInstruction::ComparisonInstruction(Opcode o,
	Comparison c, BasicBlock* b)
: BinaryInstruction(o, b), comparison(c)
{

}

Instruction* ComparisonInstruction::clone() const
{
	return ComparisonInstruction(*this);
}

Add::Add(BasicBlock* b)
: BinaryInstruction(Add, b)
{

}

Instruction* Add::clone() const
{
	return Add(*this);
}

/*! \brief An and instruction */
And::And(BasicBlock* b)
: BinaryInstruction(And, b)
{

}

Instruction* And::clone() const
{
	return And(*this);
}

/*! \brief Perform arithmetic shift right */
Ashr::Ashr(BasicBlock* b)
: Ashr(Ashr, b)
{

}

Instruction* Ashr::clone() const
{
	return Ashr(*this);
}

/*! \brief An atomic operation instruction */
Atom::Atom(Operation o, BasicBlock* b)
: BinaryInstruction(o, b), c(0)
{
	reads.push_back(c);
}

Atom::Atom(const Atom& i)
: BinaryInstruction(i), c(0)
{
	reads.push_back(c);
}

Atom& Atom::operator=(const Atom& i)
{
	if(&i == this) return *this;
	
	BinaryInstruction::operator=(i)
	
	c = reads.back();
	
	return *this;
}

Instruction* Atom::clone() const
{
	return Atom(*this);
}

/*! \brief Perform a thread group barrier */
Bar::Bar(BasicBlock* b)
: Instruction(Bar, b)
{

}

Instruction* Bar::clone() const
{
	return Bar(*this);
}

/*! \brief Perform a raw bitcast */
Bitcast::Bitcast(BasicBlock* b)
: UnaryInstruction(Bitcast, b)
{

}

Instruction* Bitcast::clone() const
{
	return Bitcast(*this);
}

/*! \brief Perform a branch */
Bra::Bra(BranchModifier m, BasicBlock* b)
: Instruction(Bra, b), target(0), modifier(m)
{

}

Bra::Bra(const Bra& i)
: Instruction(i), target(reads[1]), modifier(i.modifier)
{

}

Bra& Bra::operator=(const Bra& i)
{
	if(this == &i) return *this;
	
	Instruction::operator=(i);
	
	target   = reads[1];
	modifier = i.modifier;
	
	return *this;
}

Instruction* Bra::clone() const
{
	return Bra(*this);
}

/*! \brief Branch and save the return pc */
Call::Call(BranchModifier m, BasicBlock* b)
: Bra(m, b), link(0)
{

}

Call::Call(const Call& i)
: Bra(i), link(reads[2])
{

}

Call& Call::operator=(const Call& i)
{
	if(this == &i) return *this;
	
	Bra::operator=(i);
	
	link = reads[2];
	
	return *this;
}

Instruction* Call::clone() const
{
	return Call(*this);
}

/*! \brief Floating point division */
Fdiv::Fdiv(BasicBlock* b)
: BinaryInstruction(Fdiv, b)
{

}

Instruction* Fdiv::clone() const
{
	return Fdiv(*this);
}

/*! \brief Floating point multiplication */
Fmul::Fmul(BasicBlock* b)
: BinaryInstruction(Fmul, b)
{

}

Instruction* Fmul::clone() const
{
	return Fmul(*this);
}

/*! \brief A floating point precision extension instruction */
Fpext::Fpext(BasicBlock* b)
: UnaryInstruction(Fpext, b)
{

}

Instruction* Fpext::clone() const
{
	return Fpext(*this);
}

/*! \brief A floating point to signed integer instruction */
Fptosi::Fptosi(BasicBlock* b)
: UnaryInstruction(Fptosi, b)
{

}

Instruction* Fptosi::clone() const
{
	return Fptosi(*this);
}

/*! \brief A floating point to unsigned integer instruction */
Fptoui::Fptoui(BasicBlock* b)
: UnaryInstruction(Fptoui, b)
{

}

Instruction* Fptoui::clone() const
{
	return Fptoui(*this);
}

/*! \brief A floating point precision truncate instruction */
Fptrunc::Fptrunc(BasicBlock* b)
: UnaryInstruction(Fptrunc, b)
{

}

Instruction* Fptrunc::clone() const
{
	return Fptrunc(*this);
}

/*! \brief Floating point remainder */
Fprem::Fprem(BasicBlock* b)
: BinaryInstruction(Fprem, b)
{

}

Instruction* Fprem::clone() const
{
	return Fprem(*this);
}

/*! \brief Launch a new HTA at the specified entry point */
Launch::Launch(BasicBlock* b)
: Instruction(Launch, b)
{

}

Instruction* Launch::clone() const
{
	return Launch(*this);
}

/*! \brief Load a value from memory */
Ld::Ld(BasicBlock* b)
: UnaryInstruction(Ld, b)
{

}

Instruction* Ld::clone() const
{
	return Ld(*this);
}

/*! \brief Logical shift right */
Lshr::Lshr(BasicBlock* b)
: UnaryInstruction(Lshr, b)
{

}

Instruction* Lshr::clone() const
{
	return Lshr(*this);
}

/*! \brief Wait until memory operations at the specified level have completed */
Membar::Membar(Level l, BasicBlock* b)
: Instruction(Membar, b), level(l)
{

}

Instruction* Membar::clone() const
{
	return Membar(*this);
}

/*! \brief Multiply two operands together */
Mul::Mul(BasicBlock* b)
: BinaryInstruction(Mul, b)
{

}

Instruction* Mul::clone() const
{
	return Mul(*this);
}

/*! \brief Perform a logical OR operation */
Or::Or(BasicBlock* b)
: BinaryInstruction(Or, b)
{

}

Instruction* Or::clone() const
{
	return Or(*this);
}

/*! \brief Return from the current function call, or exit */
Ret::Ret(BasicBlock* b)
: UnaryInstruction(Ret, b)
{

}

Instruction* Ret::clone() const
{
	return Ret(*this);
}

/*! \brief Compare two operands and set a third predicate */
Setp::Setp(Comparison c, BasicBlock* b)
: ComparisonInstruction(Setp, c, b)
{

}

Instruction* Setp::clone() const
{
	return Setp(*this);
}

/*! \brief Sign extend an integer */
Sext::Sext(BasicBlock* b)
: UnaryInstruction(Sext, b)
{

}

Instruction* Sext::clone() const
{
	return Sext(*this);
}

/*! \brief Perform signed division */
Sdiv::Sdiv(BasicBlock* b)
: BinaryInstruction(Sdiv, b)
{

}

Instruction* Sdiv::clone() const
{
	return Sdiv(*this);
}

/*! \brief Perform shift left */
Shl::Shl(BasicBlock* b)
: BinaryInstriction(Shl, b)
{
	
}

Instruction* Shl::clone() const
{
	return Shl(*this);
}

/*! \brief Convert a signed int to a floating point */
Sitofp::Sitofp(BasicBlock* b)
: UnaryInstruction(Sitofp, b)
{

}

Instruction* Sitofp::clone() const
{
	return Sitofp(*this);
}

/*! \brief Perform a signed remainder operation */
Srem::Srem(BasicBlock* b)
: BinaryInstruction(Srem, b)
{

}

Instruction* Srem::clone() const
{
	return Srem(*this);
}

/*! \brief Perform a store operation */
St::St(BasicBlock* b)
: Instruction(St, b)
{
	reads.push_back(0);
	reads.push_back(0);
	
	d = reads[1];
	a = reads[2];
}

St::St(const St& s)
: Instruction(s)
{
	d = reads[1];
	a = reads[2];
}

St& St::operator=(const St& s)
{
	if(&s == this) return *this;
	
	Instruction::operator=(s);
	
	d = reads[1];
	a = reads[2];
	
	return *this;
}


Instruction* St::clone() const
{
	return St(*this);
}

/*! \brief Perform a subtract operation */
Sub::Sub(BasicBlock* b)
: BinaryInstruction(Sub, b)
{

}

Instruction* Sub::clone() const
{
	return Sub(*this);
}

/*! \brief Truncate an integer */
Trunc::Trunc(BasicBlock* b)
: BinaryInstruction(Trunc, b)
{

}

Instruction* Trunc::clone() const
{
	return Trunc(*this);
}

/*! \brief Perform an unsigned division operation */
Udiv::Udiv(BasicBlock* b)
: BinaryInstruction(Udiv, b)
{

}

Instruction* Udiv::clone() const
{
	return Udiv(*this);
}

/*! \brief Convert an unsigned int to a floating point */
Uitofp::Uitofp(BasicBlock* b)
: UnaryInstruction(Uitofp, b)
{

}

Instruction* Uitofp::clone() const
{
	return Uitofp(*this);
}

/*! \brief Perform an unsigned remainder operation */
Urem::Urem(BasicBlock* b);
Instruction* Urem::clone() const;

/*! \brief Perform a logical OR operation */
Or::Or(BasicBlock* b)
: BinaryInstruction(Or, b)
{

}

Instruction* Or::clone() const
{
	return Or(*this);
}

/*! \brief Zero extend an integer */
Zext::Zext(BasicBlock* b)
: UnaryInstruction(Zext, b)
{

}

Instruction* Zext::clone() const
{
	return Zext(*this);
}

}

}

