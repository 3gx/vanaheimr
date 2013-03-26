/*! \file   TypeAliasSet.h
	\date   March 25, 2013
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the TypeAliasSet class.
*/

#pragma once

// Forward Declarations
namespace vanaheimr { namespace ir { class Type; } }

namespace vanaheimr
{

namespace parser
{

/*! \brief An interface to a set of types and their aliases */
class TypeAliasSet
{
public:
	/*! \brief Lookup a type with the specified name, return nullptr otherwise */
	const ir::Type* getType(const std::string& name) const;

	/*! \brief Add a new mapping, override any existing mapping. */
	void addAlias(const std::string& name, const ir::Type* type);
	

};

}

}


