/*! \file   BasicBlock.h
	\date   Friday February 10, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the BasicBlock class.
*/

#pragma once

namespace vanaheimr
{

namespace ir
{

/*! \brief A list of instructions ending with a terminator. */
class BasicBlock
{
public:
	typedef std::list<VIRInstruction> InstructionList;

	typedef InstructionList::iterator       iterator;
	typedef InstructionList::const_iterator const_iterator;

	typedef unsigned int Id;

public:
	BasicBlock(Function* f, Id i);
	~BasicBlock();
	
public:
	/*! \brief Return the terminator instruction if there is one */
	      VIRInstruction* terminator();
	/*! \brief Return the terminator instruction if there is one */
	const VIRInstruction* terminator() const;

public:
	/*! \brief Inserts a new terminator instruction, possibly replacing */
	void setTerminator(const VIRInstruction& i);

public:
	iterator       begin();
	const_iterator begin() const;

	iterator       end();
	const_iterator end() const;

public:
	bool   empty() const;
	size_t size()  const;

public:
	void push_back( const VIRInstruction& i);
	void push_front(const VIRInstruction& i);

public:
	iterator insert(iterator position, const VIRInstruction& i);

private:
	Function*       _function;
	InstructionList _instructions;
	Id              _id;
};

}

}

