/*! \file   Type.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Thursday February 9, 2012
	\brief  The header file for the Type class.
	
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

// Forward Declarations
namespace vanaheimr { namespace compiler { class Compiler; } }

namespace vanaheimr
{

namespace ir
{

/*! An arbitrary Vanaheimr type */
class Type
{
public:
	typedef std::vector<const Type*> TypeVector;
	typedef compiler::Compiler       Compiler;
	typedef unsigned int             Id;
	
public:
	Type(const std::string& name, Compiler* compiler);

public:
	const std::string& name() const;
	Id                 id()   const;

public:
	bool isPrimitive()            const;
	bool isInteger()              const;
	bool isFloatingPoint()        const;
	bool isSinglePrecisionFloat() const;
	bool isDoublePrecisionFloat() const;

private:
	std::string _name;
	Id          _id;
	Compiler*   _compiler;
	
};

/*! \brief A type for an arbitrary bit-width integer */
class IntegerType : public Type
{
public:
	IntegerType(unsigned int bits);

public:
	bool isBitWidthAPowerOfTwo() const;
	unsigned int bits() const;

private:
	unsigned int _bits;
};

/*! \brief A type for an IEEE compliant 32-bit floating point type */
class FloatType : public Type
{

};

/*! \brief A type for an IEEE compliant 64-bit floating point type */
class DoubleType : public Type
{

};

/*! \brief Common functionality for aggregates (structures and arrays) */
class AggregateType : public Type
{
public:
	virtual const Type*  getTypeAtIndex  (unsigned int index) const = 0;
	virtual bool         isIndexValid    (unsigned int index) const = 0;
	virtual unsigned int numberOfSubTypes(                  ) const = 0;

};

/*! \brief A type for an array of other elements */
class ArrayType : public AggregateType
{
public:
	ArrayType(const Type* t, unsigned int elementCount);

public:
	const Type*  getTypeAtIndex  (unsigned int index) const;
	bool         isIndexValid    (unsigned int index) const;
	unsigned int numberOfSubTypes(                  ) const;

public:
	unsigned int elementsInArray() const;

private:
	const Type*  _pointedToType;
	unsigned int _elementCount;
};

/*! \brief A type for an arbitrary aggregation of types */
class StructureType : public AggregateType
{
public:
	StructureType(const TypeVector& types);

public:
	const Type*  getTypeAtIndex  (unsigned int index) const;
	bool         isIndexValid    (unsigned int index) const;
	unsigned int numberOfSubTypes(                  ) const;

private:
	TypeVector _types;

};

/*! \brief A type for a pointer */
class PointerType : public AggregateType
{
public:
	PointerType(const Type* );

public:
	const Type*  getTypeAtIndex  (unsigned int index) const;
	bool         isIndexValid    (unsigned int index) const;
	unsigned int numberOfSubTypes(                  ) const;

private:
	const Type* _pointedToType;

};

/*! \brief A type for a function */
class FunctionType : public Type
{
public:
	typedef TypeVector::const_iterator iterator;

public:
	FunctionType(const Type* returnType, const TypeVector& argumentTypes);

public:
	const Type*  getTypeAtIndex  (unsigned int index) const;
	bool         isIndexValid    (unsigned int index) const;
	unsigned int numberOfSubTypes(                  ) const;
	
public:
	iterator begin() const;
	iterator end()   const;

private:
	const Type* _returnType;
	TypeVector  _argumentTypes;
};

}

}

