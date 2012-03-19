/*	\file   VirtualRegister.cpp
	\date   Thursday March 1, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the Operand class.
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/VirtualRegister.h>

namespace vanaheimr
{

namespace ir
{

VirtualRegister::VirtualRegister(const std::string& n, Id i,
	Function* f, const Type* t)
: name(n), id(i), function(f), type(t)
{

}

}

}


