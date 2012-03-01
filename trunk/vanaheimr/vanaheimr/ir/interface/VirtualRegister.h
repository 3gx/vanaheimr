/*	\file   VirtualRegister.h
	\date   Thursday March 1, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Operand class.
*/

#pragma once

namespace vanaheimr
{

namespace ir
{

/*! \brief A virtual register in the vanaheimr IR */
class VirtualRegister
{
public:
	typedef unsigned int Id;

public:
	VirtualRegister(const std::string& name, Id id,
		Function* function, const Type* t);

public:
	std::string name;
	Id          id;
	Function*   function;
	const Type* type;
};

}

}


