/*! \file   Lexer.cpp
	\date   April 9, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Lexer class.
*/

// Vanaheimr Includes
#include <vanaheimr/parser/interface/Lexer.h>

namespace vanaheimr
{

namespace parser
{

class LexerEngine
{
public:
	class LexerContext
	{
	public:
		LexerContext(size_t position, size_t line, size_t column);
	
	public:
		size_t position;
		size_t line;
		size_t column;	
	};

public:
	std::istream* stream;

	size_t line;
	size_t column;

	LexerContextVector checkpoints;

public:
	Lexer::StringList tokenRules;
	Lexer::StringList whiteSpaceRules;

public:
	void getSimpleToken(std::string& result);
	bool getComplexToken(std::string& result);

	bool lexRegex(std::string& result, const std::string& expression);

	char snext(std::istream& stream);
};


Lexer::Lexer()
: _engine(new LexerEngine)
{

}

Lexer::~Lexer()
{
	delete _engine;
}

void Lexer::setStream(std::istream* stream)
{
	_engine->stream = stream;
	_engine->reset();
}	

std::string Lexer::peek()
{
	size_t position = _engine->stream->tellg();
	
	size_t line   = _engine->line;
	size_t column = _engine->column;
	
	auto result = nextToken();
	
	_engine->stream->seekg(position);
	
	_engine->line   = line;
	_engine->column = column;
	
	return result;
}

std::string Lexer::location() const
{
	std::stringstream stream;
	
	stream << "(" << _engine->line << ":" << _engine->column << ")";
	
	return stream.str();
}

std::string Lexer::nextToken()
{
	while(_engine->stream->good() && _engine->isWhitespace(_engine->snext()));
	_engine->stream->unget(); --_engine->column;
	
	std::string result;

	if(!_engine->getComplexToken(result))
	{
		_engine->getSimpleToken(result);
	}

	hydrazine::log("Lexer") << "scanned token '" << result << "'\n";

	return result;
}

std::string Lexer::getLine()
{
	std::string result;

	while(_engine->stream->good())
	{
		char next = _engine->snext();
		
		if(next == '\n') break;
		
		result += next;
	}

	hydrazine::log("Lexer") << "scanned line '" << result << "'\n";

	return result;
}

bool Lexer::scan(const std::string& token)
{
	hydrazine::log("Lexer") << "scanning for token '" << token << "'\n";
	
	return nextToken(stream) == token;
}

void Lexer::scanThrow(const std::string& token)
{
	if(!scan(token))
	{
		throw std::runtime_error(_location() + ": expecting a '" + token + "'");
	}
}

bool Lexer::scanPeek(const std::string& token)
{
	hydrazine::log("Lexer") << "scanning/peek for token '" << token << "'\n";
	
	return peek() == token;
}

void Lexer::reset()
{
	_engine->stream->clear();
	_engine->stream->seekg(_engine->position, std::ios::beg);
	
	_engine->line   = 0;
	_engine->column = 0;
	
	_engine->checkpoints.clear();
}

void Lexer::checkpoint()
{
	_engine->checkpoints.push_back(LexerEngine::LexerContext(_engine->stream.tellg(),
		_engine->line, _engine->column));
}

void Lexer::restoreCheckpoint()
{
	auto& checkpoints = _engine->checkpoints;

	assert(!checkpoints.empty());

	_engine->stream.clear();
	_engine->stream.seekg(checkpoints.back().position, std::ios::beg);
	
	_engine->line   = checkpoints.back().line;
	_engine->column = checkpoints.back().column;

	checkpoints.pop_back();
}

void Lexer::discardCheckpoint()
{
	assert(!_engine->checkpoints.empty());

	_engine->checkpoints.pop_back();
}

void Lexer::addTokenRegex(const std::string& regex)
{
	_engine->addRegex(regex);
}

void Lexer::addWhitespace(const std::string& whitespaceCharacters)
{
	_engine->whitespaceRules.push_back(whitespaceCharacters);
}

void Lexer::addTokens(const StringList& regexes)
{
	_engine->tokenRules.insert(_engine->tokenRules.end(), regexes.begin(),
		regexes.end());
}
	
void LexerEngine::getSimpleToken(std::string& result)
{

}

bool LexerEngine::getComplexToken(std::string& result)
{

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

bool LexerEngine::lexRegex(std::string& result, const std::string& expression);

char LexerEngine::snext(std::istream& stream);

}

}


