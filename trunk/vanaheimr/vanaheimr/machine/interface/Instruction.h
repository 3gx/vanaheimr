/*! \file   Instruction.h
	\date   Wednesday February 1, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the Operation class.

	The best documentation is clear, understandable code.
*/

#pragma once

// Standard Library Includes
#include <vector>
#include <list>

// Forward Declarations
namespace vanaheimr { namespace machine { class Operand;    } }
namespace vanaheimr { namespace machine { class BasicBlock; } }
namespace vanaheimr { namespace machine { class Operation;  } }
namespace vanaheimr { namespace machine { class MetaData;   } }

namespace vanaheimr
{

namespace machine
{

/*! \brief A model for an abstract machine instruction */
class Instruction
{
public:
	typedef Operand* OperandPointer;
	typedef std::vector<OperandPointer> OperandVector;

	typedef OperandVector::iterator       iterator;
	typedef OperandVector::const_iterator const_iterator;
		
public:
	Instruction(const Operations* op, BasicBlock* block);
	~Instruction();
	
public:
	Instruction& operator=(const Instruction&);
	Instruction(const Instruction&);
	
public:
	bool isBranch()                 const;
	bool isReturn()                 const;
	bool isLoad()                   const;
	bool isStore()                  const;
	bool isVectorLoad()             const;
	bool isVectorStore()            const;
	bool hasImmediate()             const;
	bool isBarrier()                const;
	bool isFence()                  const;
	bool isCall()                   const;
	bool isAtom()                   const;
	bool isTexture()                const;
	bool explicitlyHasSideEffects() const;
	bool isPhi()                    const;
	bool isPsi()                    const;

public:
	bool hasAttribute(const std::string& attribute) const;

public:
	iterator       read_begin();
	const_iterator read_begin() const;

	iterator       read_end();
	const_iterator read_end() const;

	iterator       write_begin();
	const_iterator write_begin() const;

	iterator       write_end();
	const_iterator write_end() const;

public:
	void output(std::ostream& stream) const;
	std::string toString() const;

public:
	OperandVector  reads;
	OperandVector  writes;
	
	OperandPointer guard;

public:
	Block* basicBlock;

private:
	 /*! \brief The machine operation performed by the instruction. */
	const Operation* _operation;

private:
	// debugging
	MetaData* _metadata;

};

}

}


