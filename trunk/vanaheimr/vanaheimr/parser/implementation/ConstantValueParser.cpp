/*! \file   TypeParser.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Thursday September 13, 2012
	\brief  The source file for the TypeParser class.
*/

// Vanaheimr Includes
#include <vanaheimr/parser/interface/ConstantValueParser.h>

#include <vanaheimr/compiler/interface/Compiler.h>

#include <vanaheimr/ir/interface/Type.h>
#include <vanaheimr/ir/interface/Constant.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <stdexcept>

namespace vanaheimr
{

namespace parser
{


ConstantValueParser::ConstantValueParser()
: _parsedConstant(nullptr)
{

}

ConstantValueParser::~ConstantValueParser()
{
	delete _parsedConstant;
}

void ConstantValueParser::parse(std::istream& stream)
{
	delete _parsedConstant;
	
	_parsedConstant = _parseConstant(stream);
}

void ConstantValueParser::parse(const ir::Type* type, std::istream& stream)
{
	delete _parsedConstant;

	if(type->isInteger() || type->isFloatingPoint())
	{	
		_parsedConstant = _parseConstant(stream);
	}
	else
	{
		_parsedConstant = _parseConstant(type, stream);
	}
}

const ir::Constant* ConstantValueParser::parsedConstant() const
{
	return _parsedConstant;
}

static bool isNumeric(char c)
{
	return c == '0' || c == '2' || c == '3' || c == '4' || c == '5' ||
		c == '6' || c == '7' || c == '8' || c == '9' || c == '1';
}

static bool isInteger(const std::string& integer)
{
	for(auto character : integer)
	{
		if(!isNumeric(character)) return false;
	}
	
	return true;
}

static bool isString(const std::string& string)
{
	for(auto character : string)
	{
		if(isNumeric(character)) return false;
	}
	
	return true;
}

static bool isFloatingPoint(const std::string& token)
{
	return !token.empty() && isNumeric(token[0]) && !isInteger(token);
}

ir::Constant* ConstantValueParser::_parseConstant(std::istream& stream)
{
	std::string nextToken = _peek(stream);
	
	ir::Constant* constant = nullptr;
	
	if(isInteger(nextToken))
	{
		constant = _parseIntegerConstant(stream);
	}
	else if(isFloatingPoint(nextToken))
	{
		constant = _parseFloatingPointConstant(stream);
	}
	else if(isString(nextToken))
	{
		constant = _parseStringConstant(stream);
	}
	
	if(constant == nullptr)
	{
		throw std::runtime_error("Failed to parse constant.");
	}

	hydrazine::log("ConstantValueParser::Parser") << "Parsed constant with type '"
		<< constant->type()->name << "'\n";
	
	return constant;
}

static ir::Constant* createZeroInitializer(const ir::Type* type)
{
	if(type->isInteger())
	{
		auto integer = static_cast<const ir::IntegerType*>(type);

		return new ir::IntegerConstant(0, integer->bits());
	}
	else if(type->isSinglePrecisionFloat())
	{
		return new ir::FloatingPointConstant(0.0f);
	}
	else if(type->isDoublePrecisionFloat())
	{
		return new ir::FloatingPointConstant(0.0);
	}
	else if(type->isStructure())
	{
		auto structure = new ir::StructureConstant(type);

		return structure;
	}

	assertM(false, "Zero initializer not implemented for " << type->name);

	return nullptr;
}

static bool isZeroInitializer(const std::string& token)
{
	return token == "zeroinitializer";
}

ir::Constant* ConstantValueParser::_parseConstant(const ir::Type* type,
	std::istream& stream)
{
	auto nextToken = _peek(stream);

	ir::Constant* constant = nullptr;

	if(isZeroInitializer(nextToken))
	{
		_nextToken(stream);
		constant = createZeroInitializer(type);
	}
	
	if(constant == nullptr)
	{
		throw std::runtime_error("Failed to parse constant.");
	}
	
	hydrazine::log("ConstantValueParser::Parser") << "Parsed constant with type '"
		<< constant->type()->name << "'\n";
	
	return constant;
}

static unsigned int parseInteger(const std::string& integer)
{
	std::stringstream stream(integer);
	
	unsigned int value = 0;
	
	stream >> value;
	
	hydrazine::log("ConstantValueParser::Parser") << " parsed integer constant '"
		<< value << "'\n";
	
	return value;
}

ir::Constant* ConstantValueParser::_parseIntegerConstant(std::istream& stream)
{
	return new ir::IntegerConstant(parseInteger(_nextToken(stream)));
}

static double parseFloat(const std::string& floating)
{
	std::stringstream stream(floating);
	
	double value = 0.0;
	
	stream >> value;
	
	hydrazine::log("ConstantValueParser::Parser") << " parsed float constant '"
		<< value << "'\n";
	
	return value;
}

ir::Constant* ConstantValueParser::_parseFloatingPointConstant(
	std::istream& stream)
{
	return new ir::FloatingPointConstant(parseFloat(_nextToken(stream)));
}

ir::Constant* ConstantValueParser::_parseStringConstant(
	std::istream& stream)
{
	std::string token = _nextToken(stream);

	hydrazine::log("ConstantValueParser::Parser") << " parsed string constant '"
		<< token << "'\n";
	
	return new ir::ArrayConstant(token.c_str(), token.size(),
		compiler::Compiler::getSingleton()->getType("i8"));
}

static bool isWhitespace(char c)
{
	return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '"';
}

static bool isToken(char c)
{
	return c == '(' || c == ')' || c == ',' || c == '[' || c == ']';
}

std::string ConstantValueParser::_peek(std::istream& stream)
{
	size_t position = stream.tellg();
	
	std::string result = _nextToken(stream);
	
	stream.clear();
	stream.seekg(position);
	
	return result;
}

std::string ConstantValueParser::_nextToken(std::istream& stream)
{
	while(stream.good() && isWhitespace(_snext(stream)));
	stream.unget();
	
	std::string result;
	
	while(stream.good() && !isWhitespace(stream.peek()))
	{
		if(!result.empty() && isToken(stream.peek())) break;
		if(!stream.good()) break;
	
		result.push_back(_snext(stream));
		
		if(isToken(*result.rbegin())) break;
	}
	
	hydrazine::log("ConstantValueParser::Lexer") << "Scanned token '"
		<< result << "'\n";

	return result;
}

bool ConstantValueParser::_scan(const std::string& token, std::istream& stream)
{
	return _nextToken(stream) == token;
}

char ConstantValueParser::_snext(std::istream& stream)
{
	char c = stream.get();
		
	return c;
}

}

}


