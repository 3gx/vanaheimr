/*! \file   TypeParser.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Thursday September 13, 2012
	\brief  The header file for the TypeParser class.
*/

#pragma once

// Standard Library Includes
#include <istream>

// Forward Declarations
namespace vanaheimr { namespace compiler { class Compiler; } }
namespace vanaheimr { namespace ir       { class Type; } }

namespace vanaheimr
{

namespace parser
{

/*! \brief A class for parsing a type from a string */
class TypeParser
{
public:
	typedef compiler::Compiler Compiler;
	
public:
	TypeParser(Compiler* c);
	~TypeParser();

public:
	            TypeParser(const TypeParser&) = delete;
	TypeParser&  operator=(const TypeParser&) = delete;

public:
	void parse(std::istream& stream);

public:
	const ir::Type* parsedType() const;

private:
	ir::Type* _parseType(std::istream& stream);

	ir::Type* _parseFunction(std::istream& stream);
	ir::Type* _parseStructure(std::istream& stream);
	ir::Type* _parseArray(const ir::Type* base, std::istream& stream);
	ir::Type* _parsePrimitive(std::istream& stream);

private:
	// Parser methods
	std::string _peek(std::istream& stream);
	
	bool _parse(std::string& token, std::istream& stream);

private:
	// Lexer methods
	std::string _nextToken(std::istream& stream);
	bool        _scan(const std::string& token, std::istream& stream);
	char        _snext(std::istream& stream);

private:
	Compiler* _compiler;
	ir::Type* _parsedType;
};

}

}


