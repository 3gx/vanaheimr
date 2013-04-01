/*! \file   LLVMParser.cpp
	\date   March 3, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the LLVM parser class.
*/

// Vanaheimr Includes
#include <vanaheimr/parser/interface/LLVMParser.h>

#include <vanaheimr/parser/interface/TypeParser.h>
#include <vanaheimr/parser/interface/TypeAliasSet.h>

#include <vanaheimr/compiler/interface/Compiler.h>

#include <vanaheimr/ir/interface/Module.h>
#include <vanaheimr/ir/interface/Type.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace vanaheimr
{

namespace parser
{

LLVMParser::LLVMParser(compiler::Compiler* compiler)
: _compiler(compiler) 
{

}

typedef compiler::Compiler Compiler;

typedef ir::Module       Module;
typedef ir::Function     Function;
typedef ir::BasicBlock   BasicBlock;
typedef ir::Type         Type;
typedef ir::Variable     Variable;
typedef ir::Global       Global;
typedef ir::FunctionType FunctionType;

class LLVMParserEngine
{
public:
	LLVMParserEngine(compiler::Compiler* compiler, const std::string& filename);

public:
	void parse(std::istream& stream);

public:
	std::string moduleName;

private:
	void _parseTypedefs(std::istream& stream);

	void _parseTopLevelDeclaration(const std::string& declaration,
		std::istream& stream);
	
	void _parseGlobalVariable(std::istream& stream);
	void _parseTypedef(std::istream& stream);
	void _parseFunction(std::istream& stream);
	void _parsePrototype(std::istream& stream);
	void _parseTarget(std::istream& stream);
	void _parseMetadata(std::istream& stream);

private:
	typedef std::set<ir::Type*> TypeSet;

private:
	void _resolveTypeAliases();
	void _resolveTypeAlias(const std::string&);
	void _resolveTypeAliasesInSubtypes(ir::Type* type, TypeSet& visited);
	
	void _parseGlobalAttributes(std::istream& stream);
	
	const Type* _parseType(std::istream& stream);
	void _addTypeAlias(const std::string& alias, const Type*);
	
	void _parseFunctionAttributes(std::istream& stream);
	void _parseFunctionBody(std::istream& stream);
	
public:
	void _resetParser(std::istream& stream);
	
private:
	// Lexer Interface
	std::string _peek(std::istream& stream);
	std::string _location() const;
	std::string _nextToken(std::istream& stream);
	std::string _getLine(std::istream& stream);
	
	void _getSimpleToken(std::string& result, std::istream& stream);
	bool _getComplexToken(std::string& result, std::istream& stream);
	std::string _getTypeString(std::istream& stream);

	bool _lexRegex(std::string& result, const std::string& expression,
		std::istream& stream);
	
	bool _scan(const std::string& token, std::istream& stream);
	void _scanThrow(const std::string& token, std::istream& stream);
	bool _scanPeek(const std::string& token, std::istream& stream);
	char _snext(std::istream& stream);
	
	void _resetLexer(std::istream& stream);
	void _checkpointLexer(std::istream& stream);
	void _restoreLexer(std::istream& stream);
	void _discardCheckpoint();

private:
	class LexerContext
	{
	public:
		LexerContext(size_t position, unsigned int line, unsigned column);
	
	public:
		size_t       position;
		unsigned int line;
		unsigned int column;
		
	};
	
	typedef std::vector<LexerContext> LexerContextVector;
	typedef std::unordered_map<std::string, std::string> StringMap;

private:
	// Parser Working State
	Compiler*   _compiler;
	Module*     _module;
	Function*   _function;
	BasicBlock* _block;

	TypeAliasSet _typedefs;
	StringMap    _typedefStrings;

private:
	// Lexer Working State
	unsigned int _line;
	unsigned int _column;

	LexerContextVector _checkpoints;
};

void LLVMParser::parse(const std::string& filename)
{
	std::ifstream file(filename.c_str());
	
	if(!file.is_open())
	{
		throw std::runtime_error("LLVM Parser: Could not open file '" +
			filename + "' for reading.");
	}

	LLVMParserEngine engine(_compiler, filename);

	engine.parse(file);

	_moduleName = engine.moduleName;
}

std::string LLVMParser::getParsedModuleName() const
{
	return _moduleName;
}

LLVMParserEngine::LLVMParserEngine(Compiler* compiler,
	const std::string& filename)
: moduleName(filename), _compiler(compiler), _line(0), _column(0)
{

}

static bool isTopLevelDeclaration(const std::string& token)
{
	return token == "@" || token == "define" || token == "declare" ||
		token == "!" || token == "target" || token == "%";
}

void LLVMParserEngine::parse(std::istream& stream)
{
	_module = &*_compiler->newModule(moduleName);

	_resetParser(stream);

	_parseTypedefs(stream);

	auto token = _nextToken(stream);

	while(isTopLevelDeclaration(token))
	{
		_parseTopLevelDeclaration(token, stream);
	
		token = _nextToken(stream);
	}

	if(!stream.eof())
	{
		throw std::runtime_error("At " + _location() +
			": hit invalid top level declaration '" + token + "'" );
	}
}

void LLVMParserEngine::_parseTypedefs(std::istream& stream)
{
	hydrazine::log("LLVM::Parser") << "Parsing typedefs\n";
	
	while(stream.good())
	{
		auto token = _nextToken(stream);

		if(token != "%") continue;
		
		auto name = _nextToken(stream);

		if(!_scan("=", stream)) continue;

		if(!_scan("type", stream)) continue;

		hydrazine::log("LLVM::Parser") << " Parsed '" << name << "'\n";
		
		_typedefStrings[name] = _getTypeString(stream);
	}

	_resolveTypeAliases();

	_resetLexer(stream);
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

static bool isLinkage(const std::string& token)
{
	return token == "private" ||
		token == "linker_private" ||
		token == "linker_private_weak" ||
		token == "internal" ||
		token == "available_externally" ||
		token == "linkonce" ||
		token == "weak" ||
		token == "common" ||
		token == "appending" ||
		token == "extern_weak" ||
		token == "linkonce_odr" ||
		token == "weak_odr" ||
		token == "linkonce_odr_auto_hide" ||
		token == "external" ||
		token == "dllimport" ||
		token == "dllexport";
}

static Variable::Linkage translateLinkage(const std::string& token)
{
	assertM(false, "Not implemented.");

	return Variable::ExternalLinkage;
}

void LLVMParserEngine::_resolveTypeAliases()
{
	hydrazine::log("LLVM::Parser") << "Initializing typedefs before "
		"parsing the remainder.\n";
	
	for(auto alias : _typedefStrings)
	{
		hydrazine::log("LLVM::Parser") << " Parsing type '"
			<< alias.first << "' with aliases.\n";
		
		TypeParser parser(_compiler, &_typedefs);
			
		std::stringstream typeStream(alias.second);
			
		parser.parse(typeStream);

		auto parsedType = *_compiler->getOrInsertType(*parser.parsedType());

		_addTypeAlias(alias.first, parsedType);
	}
	
	for(auto alias : _typedefStrings)
	{
		_resolveTypeAlias(alias.first);
	}
}

void LLVMParserEngine::_resolveTypeAlias(const std::string& alias)
{
	hydrazine::log("LLVM::Parser") << " Resolving type aliases in '"
		<< alias << "'.\n";
	
	auto aliasType = *_compiler->getOrInsertType(*_typedefs.getType(alias));

	if(aliasType == nullptr)
	{
		throw std::runtime_error("Could not find typedef entry for '" +
			alias + "'.");
	}

	TypeSet visited;

	_resolveTypeAliasesInSubtypes(aliasType, visited);
}

void LLVMParserEngine::_resolveTypeAliasesInSubtypes(
	ir::Type* type, TypeSet& visited)
{
	if(!visited.insert(type).second) return;
	
	if( type->isAlias())     return;
	if(!type->isAggregate()) return;
	
	hydrazine::log("LLVM::Parser") << "  Resolving type aliases in subtype '"
		<< type->name << "'.\n";
	
	auto aggregate = static_cast<ir::AggregateType*>(type);
	
	for(unsigned int i = 0; i < aggregate->numberOfSubTypes(); ++i)
	{
		auto subtype = aggregate->getTypeAtIndex(i);
		
		if(!subtype->isAlias())
		{
			auto originalSubtype = *_compiler->getOrInsertType(*subtype);
			_resolveTypeAliasesInSubtypes(originalSubtype, visited);
			continue;
		}
		
		auto unaliasedType = _typedefs.getType(subtype->name);
		
		if(unaliasedType == nullptr)
		{
			throw std::runtime_error("Could not find typedef entry for '" +
				subtype->name + "'.");
		}
		
		aggregate->getTypeAtIndex(i) = unaliasedType;
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
		_nextToken(stream);
	}
	else
	{
		linkage = "";
	}
	
	_parseGlobalAttributes(stream);

	auto type = _parseType(stream);

	_module->newGlobal(name, type, translateLinkage(linkage), Global::Shared);
	
	//_parseInitializer(stream);
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

	auto type = _parseType(stream);

	_addTypeAlias(name, type);
}

void LLVMParserEngine::_parseFunction(std::istream& stream)
{
	 _parsePrototype(stream);
	_parseFunctionAttributes(stream);

	_scanThrow("{", stream);

	_parseFunctionBody(stream);

	_scanThrow("}", stream);

}

void LLVMParserEngine::_parsePrototype(std::istream& stream)
{
	auto returnType = _parseType(stream);

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
			
			auto next = _peek(stream);

			if(next != ",") break;
			
			_scan(",", stream);
		}
		
	}

	_scanThrow(")", stream);

	auto type = _compiler->getOrInsertType(FunctionType(_compiler,
		returnType, argumentTypes));

	_function = &*_module->newFunction(name, Variable::ExternalLinkage,
		Variable::HiddenVisibility, *type);
}

void LLVMParserEngine::_parseTarget(std::istream& stream)
{
	hydrazine::log("LLVM::Parser") << "Parsing target\n";

	auto name = _nextToken(stream);

	_scanThrow("=", stream);

	auto targetString = _nextToken(stream);

	hydrazine::log("LLVM::Parser") << " target:'" << name << " = "
		<< targetString << "'\n";

	// TODO: use this
}

void LLVMParserEngine::_parseMetadata(std::istream& stream)
{
	hydrazine::log("LLVM::Parser") << "Parsing metadata\n";

	assertM(false, "Not Implemented.");
}

void LLVMParserEngine::_parseGlobalAttributes(std::istream& stream)
{
	assertM(false, "Not Implemented.");
}

const Type* LLVMParserEngine::_parseType(std::istream& stream)
{
	TypeParser parser(_compiler, &_typedefs);
	
	parser.parse(stream);
	
	hydrazine::log("LLVM::Parser") << "Parsed type '"
		<< parser.parsedType()->name << "'\n";
	
	return parser.parsedType();
}

void LLVMParserEngine::_addTypeAlias(const std::string& alias, const Type* type)
{
	hydrazine::log("LLVM::Parser") << " alias '" << alias << "' -> '"
		<< type->name << "'\n";

	_typedefs.addAlias(alias, type);
}

void LLVMParserEngine::_parseFunctionAttributes(std::istream& stream)
{
	assertM(false, "Not Implemented.");
}

void LLVMParserEngine::_parseFunctionBody(std::istream& stream)
{
	assertM(false, "Not Implemented.");
}

void LLVMParserEngine::_resetParser(std::istream& stream)
{
	_resetLexer(stream);
	
	_typedefs.clear();
	_typedefStrings.clear();
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
	return c == ' ' || c == '\n' || c == '\r' || c == '\t';
}

static bool isToken(char c)
{
	return c == '|' || c == '(' || c == ')' || c == ';' || c == ',' || c == '='
		|| c == '%' || c == '@' || c == '[' || c == ']' || c == '*';
}

bool LLVMParserEngine::_getComplexToken(std::string& result,
	std::istream& stream)
{
	// comment: ';*\n'
	if(_lexRegex(result, ";*\n", stream))
	{
		result.clear();
		return false;
	}

	// string: '"*"'
	if(_lexRegex(result, "\"*\"", stream)) return true;

	return false;
}

void LLVMParserEngine::_getSimpleToken(std::string& result,
	std::istream& stream)
{
	// Simple tokens
	while(stream.good() && !isWhitespace(stream.peek()))
	{
		if(!result.empty() && isToken(stream.peek())) break;
	
		result.push_back(_snext(stream));
			
		if(result.size() == 1 && isToken(*result.rbegin())) break;
	}
}

std::string LLVMParserEngine::_nextToken(std::istream& stream)
{
	while(stream.good() && isWhitespace(_snext(stream)));
	stream.unget(); --_column;
	
	std::string result;

	if(!_getComplexToken(result, stream))
	{
		_getSimpleToken(result, stream);
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

std::string LLVMParserEngine::_getTypeString(std::istream& stream)
{
	auto token = _nextToken(stream);

	if(token != "{") return token;

	std::string result("{");

	unsigned int count = 1;

	while(count > 0 && stream.good())
	{
		char next = _snext(stream);
		
		if(next == '{')
		{
			++count;
		}
		else if(next == '}')
		{
			--count;
		}

		result += next;
	}
	
	hydrazine::log("LLVM::Lexer") << "scanned type string '" << result << "'\n";

	return result;
}

static bool isWildcard(char c)
{
	return c == '*';
}

static bool matchedWildcard(std::string::const_iterator next,
	const std::string& expression, char c)
{
	if(!isWildcard(*next)) return false;
	
	auto following = next; ++following;
	
	if(following == expression.end()) return false;
	
	return isWildcard(*next) && *following != c;
}

static bool regexMatch(std::string::const_iterator next,
	const std::string& expression, char c)
{
	if(isWildcard(*next)) return true;
	
	return *next == c;
}

bool LLVMParserEngine::_lexRegex(std::string& result,
	const std::string& expression, std::istream& stream)
{
	if(expression.empty()) return false;

	assert(!isWildcard(expression.back()));

	auto next = expression.begin();
	
	if(!regexMatch(next, expression, stream.peek())) return false;
	
	_checkpointLexer(stream);
	
	for(; next != expression.end(); )
	{
		if(!regexMatch(next, expression, stream.peek()))
		{
			_restoreLexer(stream);
			result.clear();
			return false;
		}
		
		result.push_back(_snext(stream));
		
		if(!matchedWildcard(next, expression, stream.peek()))
		{
			++next;
		}
	}
	
	return true;
}

bool LLVMParserEngine::_scan(const std::string& token, std::istream& stream)
{
	hydrazine::log("LLVM::Lexer") << "scanning for token '" << token << "'\n";
	
	return _nextToken(stream) == token;
}

void LLVMParserEngine::_scanThrow(const std::string& token,
	std::istream& stream)
{
	if(!_scan(token, stream))
	{
		throw std::runtime_error(_location() + ": expecting a '" + token + "'");
	}
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
	
	_checkpoints.clear();
}

void LLVMParserEngine::_checkpointLexer(std::istream& stream)
{
	_checkpoints.push_back(LexerContext(stream.tellg(), _line, _column));
}

void LLVMParserEngine::_restoreLexer(std::istream& stream)
{
	assert(!_checkpoints.empty());

	stream.clear();
	stream.seekg(_checkpoints.back().position, std::ios::beg);
	_line = _checkpoints.back().line;
	_column = _checkpoints.back().column;
	
	_checkpoints.pop_back();
}

void LLVMParserEngine::_discardCheckpoint()
{
	assert(!_checkpoints.empty());

	_checkpoints.pop_back();
}

LLVMParserEngine::LexerContext::LexerContext(size_t p,
	unsigned int l, unsigned int c)
: position(p), line(l), column(c)
{

}

}

}


