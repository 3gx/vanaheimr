/*! \file   Function.h
	\date   Friday February 3, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Function class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/ir/interface/BasicBlock.h>
#include <vanaheimr/ir/interface/Argument.h>
#include <vanaheimr/ir/interface/Value.h>

namespace vanaheimr
{

namespace ir
{

/*! \brief Describes a vanaheimr function */
class Function : public Variable
{
public:
	typedef std::list<BasicBlock> BasicBlockList;
	typedef std::list<Argument>   ArgumentList;
	typedef std::list<Value>      ValueList;

	typedef BasicBlockList::iterator       iterator;
	typedef BasicBlockList::const_iterator const_iterator;

	typedef ArgumentList::iterator       argument_iterator;
	typedef ArgumentList::const_iterator const_argument_iterator;

public:
	Function(const std::string& name = "", Module* m = 0,
		Linkage l = ExternalLinkage);
	
public:
	iterator       begin();
	const_iterator begin() const;
	
	iterator       end();
	const_iterator end() const;

public:
	size_t size()  const;
	bool   empty() const;

public:
	      BasicBlock& front();
	const BasicBlock& front() const;

	      BasicBlock& back();
	const BasicBlock& back() const;

public:
	iterator newBasicBlock(iterator position, const std::string& name);

public:
	argument_iterator       argument_begin();
	const_argument_iterator argument_begin() const;
	
	argument_iterator       argument_end();
	const_argument_iterator argument_end() const;

public:
	size_t argument_size()  const;
	bool   argument_empty() const;

public:
	value_iterator       value_begin();
	const_value_iterator value_begin() const;
	
	value_iterator       value_end();
	const_value_iterator value_end() const;

public:
	size_t value_size()  const;
	bool   value_empty() const;

private:
	BasicBlockList _lists;
	ArgumentList   _arguments;
	ValueList      _values;
};

}

}


