/*! \file   LLVMParser.cpp
	\date   March 3, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the LLVM parser class.
*/

// Vanaheimr Includes
#include <vanaheimr/parser/interface/LLVMParser.h>

// Standard Library Includes
#include <fstream>

namespace vanaheimr
{

namespace parser
{

LLVMParser::LLVMParser(compiler::Compiler* compiler)
: _compiler(compiler) 
{

}

typedef compiler::Compiler Compiler;

typedef ir::Module     Module;
typedef ir::Function   Function;
typedef ir::BasicBlock BasicBlock;

class LLVMParserEngine
{
public:
	LLVMParserEngine(compiler::Compiler* compiler, const std::string& filename);

public:
	void parse(std::istream& stream);

public:
	std::string moduleName;

private:
	void _parseTopLevelDeclaration(const std::string& declaration,
		std::istream& stream);
	
	void _parseGLo

private:
	// Parser Working State
	Compiler*   _compiler;
	Module*     _module;
	Functin*    _function;
	BasicBlock* _block;

private:
	// Lexer Working State
	unsigned int line;
	unsigned int column;

};

void LLVMParser::parse(const std::string& filename)
{
	std::ifstream file(filename.c_str());
	
	if(!file.is_open())
	{
		throw std::runtime_error("LLVM Parser: Could not open file '" +
			filename + "' for reading.");
	}

	LLVMParserEngine engine(compiler, filename);

	engine.parse(file);

	_moduleName = engine.moduleName;
}

LLVMParserEngine(compiler::Compiler* compiler, const std::string& filename);
: moduleName(filename), _compiler(compiler)
{

}

std::string LLVMParser::getParsedModuleName() const
{
	return _moduleName;
}

static bool isTopLevelDeclaration(const std::string& token)
{
	return token == "@" || token == "define" || token == "declare"
		|| token == "!";
}

void LLVMParserEngine::parse(std::istream& stream)
{
	auto token = _nextToken(stream);

	while(isTopLevelDeclaration(token))
	{
		_parseTokenLevelDeclaration(token, stream);
	
		token = _nextToken(stream);
	}

	if(!stream.eof())
	{
		throw std::runtime_error("At " + _location() +
			": hit invalid top level declaration '" + token + "'" );
	}
}

void LLVMParserEngine::_parseTopLevelDeclaration(const std::string& token,
	std::istream& stream)
{
	if(token == "@")
	{
		_parseGlobalVariable(stream);
	}
	else if(token == "define")
	{
		_parseFunction(stream);
	}
	else if(token == "declare")
	{
		_parsePrototype(stream);
	}
	else
	{
		_parseMetadata(stream);
	}
}

}

}


