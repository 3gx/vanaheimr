/*! \file   Constant.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Thursday February 9, 2012
	\brief  The source file for the Constant family of classes.
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Constant.h>

#include <vanaheimr/compiler/interface/Compiler.h>

namespace vanaheimr
{

namespace ir
{

Constant::Constant(const Type* type)
: _type(type)
{

}

Constant::~Constant()
{

}

const Type* Constant::type() const
{
	return _type;
}

ArrayConstant::ArrayConstant(const void* data, uint64_t size)
: Constant(compiler::Compiler::getSingleton()->getType("i8")),
	_value((const uint8_t*)data, (const uint8_t*)data + size)
{

}

uint64_t ArrayConstant::size() const
{
	return _value.size();
}

bool ArrayConstant::isNullValue() const
{
	for(auto value : _value)
	{
		if(value != 0) return false;
	}
	
	return true;
}

Constant::DataVector ArrayConstant::data() const
{
	return _value;
}

size_t ArrayConstant::bytes() const
{
	return size();
}

Constant* ArrayConstant::clone() const
{
	return new ArrayConstant(*this);
}

}

}

