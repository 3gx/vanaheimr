/*! \file   Compiler.h
	\date   Sunday February 12, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Compiler class.
	
*/


// Forward Declarations
namespace vanaheimr { namespace ir { class Type; } }

namespace vanaheimr
{

/*! \brief A namespace for core compiler components */
namespace compiler
{

/*! \brief The global compiler state for vanaheimr. */
class Compiler
{
public:
	typedef std::vector<Type*> TypeVector;
	
	typedef TypeVector::iterator       iterator;
	typedef TypeVector::const_iterator const_iterator;

public:
	Compiler();

public:
	iterator       begin();
	const_iterator begin() const;

	iterator       end();
	const_iterator end() const;

public:
	bool   empty() const;
	size_t size()  const;

public:
	/*! \brief Lookup a type by name, return 0 if it doesn't exist */
	Type*       getType(const std::string& name);
	/*! \brief Lookup a type by name, return 0 if it doesn't exist */
	const Type* getType(const std::string& typeName) const;

private:
	TypeVector _types;

};	

}

}


