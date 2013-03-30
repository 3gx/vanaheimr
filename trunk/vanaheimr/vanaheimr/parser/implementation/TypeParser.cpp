/*! \file   TypeParser.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Thursday September 13, 2012
	\brief  The source file for the TypeParser class.
*/

// Vanaheimr Includes
#include <vanaheimr/parser/interface/TypeParser.h>
#include <vanaheimr/parser/interface/TypeAliasSet.h>

#include <vanaheimr/compiler/interface/Compiler.h>

#include <vanaheimr/ir/interface/Type.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <stdexcept>

namespace vanaheimr
{

namespace parser
{

TypeParser::TypeParser(Compiler* c, const TypeAliasSet* a)
: _compiler(c), _parsedType(nullptr), _typedefs(a)
{

}

TypeParser::~TypeParser()
{
}

void TypeParser::parse(std::istream& stream)
{
	_parsedType = nullptr;
	
	_parsedType = _parseType(stream);
}

const ir::Type* TypeParser::parsedType() const
{
	assert(_parsedType != nullptr);
	
	return _parsedType;
}

static bool isFunction(const std::string& token)
{
	return token.find("(") == 0;
}

static bool isStructure(const std::string& token)
{
	return token.find("{") == 0;
}

static bool isArray(const std::string& token)
{
	return token.find("[") == 0;
}

static bool isPointer(const std::string& token)
{
	return token.find("*") == 0;
}

static bool isVariadic(const std::string& token)
{
	return token == "...";
}

static bool isTypeAlias(const std::string& token)
{
	return token.find("%") == 0;
}

static bool isOpaqueType(const std::string& token)
{
	return token.find("opaque") == 0;
}

static bool isPrimitive(compiler::Compiler* compiler, const std::string& token)
{
	hydrazine::log("TypeParser::Parser") << "Checking if " << token
		<< " is a primitive type.\n";
	
	ir::Type* primitive = compiler->getType(token);

	if(primitive == nullptr) return false;

	return primitive->isPrimitive() || primitive->isBasicBlock();
}

ir::Type* TypeParser::_parseType(std::istream& stream)
{
	std::string nextToken = _peek(stream);
	
	ir::Type* type = nullptr;
	
	if(isFunction(nextToken))
	{
		type = _parseFunction(stream);
	}
	else if(isStructure(nextToken))
	{
		type = _parseStructure(stream);
	}
	else if(isPrimitive(_compiler, nextToken))
	{
		type = _parsePrimitive(stream);
		
		nextToken = _peek(stream);
		
		if(isFunction(nextToken))
		{
			type = _parseFunction(type, stream);
		}
	}
	else if(isArray(nextToken))
	{
		type = _parseArray(stream);
	}
	else if(isVariadic(nextToken))
	{
		_scan("...", stream);
		type = *_compiler->getOrInsertType(ir::VariadicType(_compiler));
	}
	else if(isTypeAlias(nextToken))
	{
		type = _parseTypeAlias(stream);
	}
	else if(isOpaqueType(nextToken))
	{
		type = *_compiler->getOrInsertType(ir::OpaqueType(_compiler));
	}

	nextToken = _peek(stream);

	while(isPointer(nextToken))
	{
		_scan("*", stream);
		type = *_compiler->getOrInsertType(ir::PointerType(_compiler, type));
	
		nextToken = _peek(stream);
	}
	
	if(type == nullptr)
	{
		throw std::runtime_error("Failed to parse type.");
	}
	
	hydrazine::log("TypeParser::Parser") << "Parsed type " << type->name()
		<< ".\n";
	
	return type;
}

ir::Type* TypeParser::_parseFunction(std::istream& stream)
{
	ir::Type* returnType = nullptr;
	ir::Type::TypeVector argumentTypes;

	if(!_scan("(", stream))
	{
		throw std::runtime_error("Failed to parse function "
			"type, expecting '('.");
	}

	std::string closeBrace = _peek(stream);

	if(closeBrace != ")")
	{
		returnType = _parseType(stream);
	}

	if(!_scan(")", stream))
	{
		throw std::runtime_error("Failed to parse function "
			"type, expecting ')'.");
	}

	if(!_scan("(", stream))
	{
		throw std::runtime_error("Failed to parse function "
			"type, expecting '('.");
	}
	
	closeBrace = _peek(stream);

	if(closeBrace != ")")
	{
		do
		{
			argumentTypes.push_back(_parseType(stream));
		
			std::string comma = _peek(stream);
			
			if(comma == ",")
			{
				_scan(",", stream);
			}
			else
			{
				break;
			}
		}
		while(true);
	}

	if(!_scan(")", stream))
	{
		throw std::runtime_error("Failed to parse function "
			"type, expecting ')'.");
	}

	return *_compiler->getOrInsertType(ir::FunctionType(
		_compiler, returnType, argumentTypes));
}

ir::Type* TypeParser::_parseFunction(const ir::Type* returnType,
	std::istream& stream)
{
	ir::Type::TypeVector argumentTypes;

	if(!_scan("(", stream))
	{
		throw std::runtime_error("Failed to parse function "
			"type, expecting '('.");
	}
       
	auto closeBrace = _peek(stream);

	if(closeBrace != ")")
	{
		do
		{
			argumentTypes.push_back(_parseType(stream));
	       
			std::string comma = _peek(stream);
		       
			if(comma == ",")
			{
				_scan(",", stream);
			}
			else
			{
				break;
			}
		}
		while(true);
	}

	if(!_scan(")", stream))
	{
		throw std::runtime_error("Failed to parse function "
			"type, expecting ')'.");
	}

	return new ir::FunctionType(_compiler, returnType, argumentTypes);
}


ir::Type* TypeParser::_parseStructure(std::istream& stream)
{
	if(!_scan("{", stream))
	{
		throw std::runtime_error("Failed to parse structure "
			"type, expecting '{'.");
	}

	ir::Type::TypeVector types;

	auto closeBrace = _peek(stream);

	if(closeBrace != "}")
	{
		do
		{
			types.push_back(_parseType(stream));
		
			std::string comma = _peek(stream);
			
			if(comma == ",")
			{
				_scan(",", stream);
			}
			else
			{
				break;
			}
		}
		while(true);
	}
	
	if(!_scan("}", stream))
	{
		throw std::runtime_error("Failed to parse structure "
			"type, expecting '}'.");
	}

	return *_compiler->getOrInsertType(ir::StructureType(_compiler, types));
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

static unsigned int parseInteger(const std::string& integer)
{
	if(!isInteger(integer))
	{
		throw std::runtime_error("Failed to parse array "
			"type, expecting an integer.");
	}
	
	std::stringstream stream(integer);
	
	unsigned int value = 0;
	
	stream >> value;
	
	return value;
}

ir::Type* TypeParser::_parseArray(std::istream& stream)
{
	if(!_scan("[", stream))
	{
		throw std::runtime_error("Failed to parse array "
			"type, expecting '['.");
	}
	
	std::string dimensionToken = _nextToken(stream);
	
	unsigned int dimension = parseInteger(dimensionToken);
	
	if(!_scan("x", stream))
	{
		throw std::runtime_error("Failed to parse array "
			"type, expecting 'x'.");
	}

	auto base = _parseType(stream);
	
	if(!_scan("]", stream))
	{
		throw std::runtime_error("Failed to parse array "
			"type, expecting ']'.");
	}
	
	return *_compiler->getOrInsertType(ir::ArrayType(_compiler,
		base, dimension));
}

ir::Type* TypeParser::_parsePrimitive(std::istream& stream)
{
	std::string token;
	
	bool success = _parse(token, stream);
	
	if(!success)
	{
		throw std::runtime_error("Hit end of stream while "
			"searching for primitive type.");
	}
	
	return _compiler->getType(token);
}

ir::Type* TypeParser::_parseTypeAlias(std::istream& stream)
{
	if(!_scan("%", stream))
	{
		throw std::runtime_error("Failed to parse type alias, expecting '%'.");
	}
	
	std::string token;

	bool success = _parse(token, stream);
	
	if(!success)
	{
		throw std::runtime_error("Hit end of stream while "
			"searching for type alias type.");
	}
	
	auto alias = _getTypeAlias(token);

	if(alias == nullptr)
	{
		throw std::runtime_error("Failed to parse type alias, unknown "
			"typename '" + token + "'.");
	}

	return alias;
}

ir::Type* TypeParser::_getTypeAlias(const std::string& token)
{
	if(_typedefs == nullptr) return nullptr;

	auto type = _typedefs->getType(token);

 	if(type != nullptr) return *_compiler->getOrInsertType(*type);

	return *_compiler->getOrInsertType(ir::AliasedType(_compiler, token));
}

static bool isWhitespace(char c)
{
	return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

static bool isToken(char c)
{
	return c == '(' || c == ')' || c == ',' || c == '[' || c == ']' || c == '{' ||
		c == '}' || c == '*' || c == '%';
}

std::string TypeParser::_peek(std::istream& stream)
{
	std::string result;
	
	size_t position = stream.tellg();
	
	_parse(result, stream);
	
	stream.clear();
	stream.seekg(position);
	
	return result;
}

bool TypeParser::_parse(std::string& token, std::istream& stream)
{
	token = _nextToken(stream);

	return !token.empty();
}

std::string TypeParser::_nextToken(std::istream& stream)
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

	hydrazine::log("TypeParser::Lexer") << "scanned token '" << result << "'\n";

	return result;
}

bool TypeParser::_scan(const std::string& token, std::istream& stream)
{
	hydrazine::log("TypeParser::Lexer") << "scanning for token '" << token << "'\n";
	
	return _nextToken(stream) == token;
}

char TypeParser::_snext(std::istream& stream)
{
	char c = stream.get();
		
	return c;
}

}

}


