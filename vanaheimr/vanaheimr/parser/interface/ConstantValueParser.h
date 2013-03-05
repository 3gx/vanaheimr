/*! \file   ConstantValueParser.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Thursday March 4 13, 2013
	\brief  The header file for the ConstantValueParser class.
*/

#pragma once

// Standard Library Includes
#include <istream>

// Forward Declarations
namespace vanaheimr { namespace ir { class Constant; } }

namespace vanaheimr
{

namespace parser
{

/*! \brief A class for parsing a type from a string */
class ConstantValueParser
{
public:
	ConstantValueParser();
	~ConstantValueParser();

public:
	            ConstantValueParser(const ConstantValueParser&) = delete;
	ConstantValueParser&  operator=(const ConstantValueParser&) = delete;

public:
	void parse(std::istream& stream);

public:
	const ir::Constant* parsedConstant() const;

private:
	// Specialized Parsing
	ir::Constant* _parseConstant(std::istream& stream);

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
	ir::Constant* _parsedConstant;
};

}

}


