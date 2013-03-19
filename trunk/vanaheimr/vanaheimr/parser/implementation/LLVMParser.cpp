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
	
	void _parseGlobalVariable(std::istream& stream);
	void _parseTypedef(std::istream& stream);
	void _parseFunction(std::istream& stream);
	void _parsePrototype(std::istream& stream);
	void _parseMetadata(std::istream& stream);

private:
	std::string _peek(std::istream& stream);
	std::string _location() const;
	std::string _nextToken(std::istream& stream);
	std::string _getLine(std::istream& stream);
	bool _scan(const std::string& token, std::istream& stream);
	bool _scanPeek(const std::string& token, std::istream& stream);
	char _snext(std::istream& stream);
	void _resetLexer(std::istream& stream);

private:
	// Parser Working State
	Compiler*   _compiler;
	Module*     _module;
	Functin*    _function;
	BasicBlock* _block;

private:
	// Lexer Working State
	unsigned int _line;
	unsigned int _column;

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
		|| token == "!" || token == "target";
}

void LLVMParserEngine::parse(std::istream& stream)
{
	_module = &*_compiler->newModule(moduleName);

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
	else if(token == "%")
	{
		_parseTypedef(stream);
	}
	else if(token == "define")
	{
		_parseFunction(stream);
	}
	else if(token == "declare")
	{
		_parsePrototype(stream);
	}
	else if(token == "target")
	{
		_parseTarget(stream);
	}
	else
	{
		_parseMetadata(stream);
	}
}
	
void LLVMParserEngine::_parseGlobalVariable(std::istream& stream)
{
	auto name = _nextToken(stream);

	if(!_scan("=", stream))
	{
		throw std::runtime_error("At " + _location() + ": expecting a '='.");
	}
	
	auto linkage = _peek(stream);

	if(isLinkage(linkage))
	{
		_nextToken();
	}
	else
	{
		linkage = "";
	}
	
	_parseGlobalAttributes(stream);

	auto type = _parseType(stream);

	auto global = _module->newGlobal(name, type,
		translateLinkage(linkage), _visibility);
	
	_parseInitializer(stream);
}

void LLVMParserEngine::_parseTypedef(std::istream& stream)
{
	auto name = _nextToken(stream);
	
	if(!_scan("=", stream))
	{
		throw std::runtime_error("At " + _location() + ": expecting a '='.");
	}
	
	if(!_scan("type", stream))
	{
		throw std::runtime_error("At " + _location() + ": expecting 'type'.");
	}

	auto type = _parseType(type);

	_addTypeAlias(name, type);
}

void LLVMParserEngine::_parseFunction(std::istream& stream)
{
	 _parsePrototype(stream);
	_parseAttributes(stream);

	_scanThrow("{", stream);

	_parseFunctionBody(stream);

	_scanThrow("}", stream);

}

void LLVMParserEngine::_parsePrototype(std::istream& stream)
{
	auto returnType = _parseType(type);

	_scanThrow("@", stream);
	
	auto name = _nextToken(stream);

	_scanThrow("(", stream);

	auto end = _peek(stream);

	Type::TypeVector argumentTypes;

	if(end != ")")
	{
		while(true)
		{
			argumentTypes.push_back(_parseType(stream));
			
			auto next = _peek();

			if(next != ",") break;
			
			_scan(",", stream);
		}
		
	}

	_scanThrow(")", stream);

	auto type = &*_compiler->getOrInsertType(FunctionType(_compiler,
		returnType, argumentTypes));

	_function = &*_module->newFunction(name, Variable::ExternalLinkage,
		Variable::HiddenVisibility, type);

}

void LLVMParserEngine::_parseMetadata(std::istream& stream)
{
	hydrazine::log("LLVM:Parser:") << "Parsing metadata\n";

	assertM(false, "Not Implemented.");
}


std::string LLVMParserEngine::_peek(std::istream& stream)
{
	size_t position = stream.tellg();
	
	unsigned int line   = _line;
	unsigned int column = _column;
	
	std::string result = _nextToken(stream);
	
	stream.seekg(position);
	
	_line   = line;
	_column = column;
	
	return result;
}

std::string LLVMParserEngine::_location() const
{
	std::stringstream stream;
	
	stream << "(" << _line << ":" << _column << ")";
	
	return stream.str();
}

static bool isWhitespace(char c)
{
	return c == ' ' || c == '\n' || c == '\r' || c == '\t' || c == '"';
}

static bool isToken(char c)
{
	return c == '|' || c == '(' || c == ')' || c == ';' || c == ',' || c == '=';
}

std::string LLVMParserEngine::_nextToken(std::istream& stream)
{
	while(stream.good() && isWhitespace(_snext(stream)));
	stream.unget(); --_column;
	
	std::string result;
	
	while(stream.good() && !isWhitespace(stream.peek()))
	{
		if(!result.empty() && isToken(stream.peek())) break;
	
		result.push_back(_snext(stream));
		
		if(isToken(*result.rbegin())) break;
	}

	hydrazine::log("LLVM::Lexer") << "scanned token '" << result << "'\n";

	return result;
}

std::string LLVMParserEngine::_getLine(std::istream& stream)
{
	std::string result;

	while(stream.good())
	{
		char next = _snext(stream);
		
		if(next == '\n') break;
		
		result += next;
	}

	hydrazine::log("LLVM::Lexer") << "scanned line '" << result << "'\n";

	return result;
}

bool LLVMParserEngine::_scan(const std::string& token, std::istream& stream)
{
	hydrazine::log("LLVM::Lexer") << "scanning for token '" << token << "'\n";
	
	return _nextToken(stream) == token;
}

bool LLVMParserEngine::_scanPeek(const std::string& token, std::istream& stream)
{
	hydrazine::log("LLVM::Lexer") << "scanning/peek for token '"
		<< token << "'\n";
	
	return _peek(stream) == token;
}

char LLVMParserEngine::_snext(std::istream& stream)
{
	char c = stream.get();
	
	if(c == '\n')
	{
		++_line;
		_column = 0;
	}
	else
	{
		++_column;
	}
	
	return c;
}

void LLVMParserEngine::_resetLexer(std::istream& stream)
{
	stream.clear();
	stream.seekg(0, std::ios::beg);
	_line = 0;
	_column = 0;
}

}

}


