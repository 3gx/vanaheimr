/*! \file   Lexer.cpp
	\date   April 9, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Lexer class.
*/

// Vanaheimr Includes
#include <vanaheimr/parser/interace/Lexer.h>

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
};

	void getSimpleToken(std::string& result);
	bool getComplexToken(std::string& result);

	bool lexRegex(std::string& result, const std::string& expression, std::istream& stream);

	char snext(std::istream& stream);

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

void Lexer::checkpoint();
void Lexer::restoreCheckpoint();
void Lexer::discardCheckpoint();

void Lexer::addTokenRegex(const std::string& regex);
void Lexer::addWhitespace(const std::string& whitespaceCharacters);	
void Lexer::addTokens(const StringList& regexes);

}

}


