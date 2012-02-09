/*! \file   Constant.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Thursday February 9, 2012
	\brief  The header file for the Constant family of classes.
*/

namespace vanaheimr
{

namespace ir
{

/*! \brief A base class for any constant */
class Constant
{
public:
	Constant(const Type* type);

public:
	virtual bool isNullValue() const;
	
public:
	const Type* type() const;
	
private:
	const Type* _type;
};

/*! \brief Floating point data */
class FloatingPointConstant : public Constant
{
public:
	FloatingPointConstant(float f);
	FloatingPointConstant(double d);

public:
	operator float();
	operator double();

public:
	FloatingPointConstant(const FloatingPointConstant& f);
	FloatingPointConstant& operator=(const FloatingPointConstant& f);

private:
	union
	{
		float  _float;
		double _double;
	};

};

/*! \brief Integer data */
class IntegerConstant : public Constant
{
public:
	IntegerConstant(uint64_t i);	

public:
	operator uint64_t() const;
	operator int64_t()  const;
	operator uint32_t() const;
	operator int32_t()  const;
	
public:
	IntegerConstant& operator=(uint64_t i);

private:
	uint64_t _value;
};

/*! \brief A pointer constant */
class PointerConstant : public Constant
{
public:
	PointerConstant(uint64_t i);	
	PointerConstant(void* i);	

public:
	operator uint64_t() const;
	operator int64_t()  const;
	operator uint32_t() const;
	operator int32_t()  const;
	operator void*()    const;
	
public:
	PointerConstant& operator=(uint64_t i);
	PointerConstant& operator=(void*    i);

private:
	uint64_t _pointer;
};

/*! \brief A structure constant */
class StructureConstant : public Constant
{
public:
	StructureConstant(const Type* aggregateType);

public:
	Constant*       getMember(unsigned int index);
	const Constant* getMember(unsigned int index) const;

private:
	typedef std::vector<Constant*> ConstantVector;

private:
	ConstantVector _members;
};

/*! \brief An array constant for normal data types */
class ArrayConstant : public Constant
{
public:
	ArrayConstant(const void* data, uint64_t size);

public:
	void*    data() const;
	uint64_t size() const;
	
private:
	typedef std::vector<uint8_t> DataVector;
	
private:
	DataArray _value;
};

/*! \brief An array constant */
class ComplexArrayConstant : public StructureConstant
{
public:
	ComplexArrayConstant(const Type* t);

};

/*! \brief An address of a basic block */
class BasicBlockAddressConstant : public Constant
{
public:
	BasicBlockAddressConstant(const BasicBlock* block);

public:
	operator const BasicBlock*() const;
	
public:
	PointerConstant& operator=(const BasicBlock* b);

private:
	const BasicBlock* _basicBlock;
};

}

}

