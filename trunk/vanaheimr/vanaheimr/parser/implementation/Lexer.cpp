/*! \file   Lexer.cpp
	\date   April 9, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Lexer class.
*/

// Vanaheimr Includes
#include <vanaheimr/parser/interface/Lexer.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <vector>
#include <sstream>
#include <cassert>
#include <stdexcept>
#include <set>

namespace vanaheimr
{

namespace parser
{

class LexerEngine
{
public:
	typedef std::set<std::string*> RuleSet;

	class TokenDescriptor
	{
	public:
		explicit TokenDescriptor(LexerEngine* engine);
	
	public:
		size_t beginPosition;
		size_t endPosition;
	
	public:
		size_t line;
		size_t column;
		
	public:
		RuleSet possibleMatches;
	};
	
	typedef std::vector<TokenDescriptor> TokenVector;
	typedef TokenVector::iterator LexerContext;

	typedef std::vector<LexerContext> LexerContextVector;

public:
	std::istream* stream;

	size_t line;
	size_t column;

	LexerContextVector checkpoints;

public:
	Lexer::StringList tokenRules;
	Lexer::StringList whitespaceRules;

public:
	std::string nextToken();
	std::string peek();

public:
	void reset(std::istream* s);

	void checkpoint();
	void restore();

private:
	TokenVector           _tokens;
	TokenVector::iterator _nextToken;

private:
	void _createTokens();
	void _mergeTokens();
	
	char _snext();

private:
	TokenDescriptor _mergeWithBegin(const TokenDescriptor& token);
	TokenDescriptor _mergeWithEnd(const TokenDescriptor& token));
	TokenDescriptor _mergeWithNext(const TokenDescriptor& token,
		const TokenDescriptor& next);
		
	TokenDescriptor _canMerge(const TokenDescriptor& token,
		const TokenDescriptor& next);
	
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
	_engine->reset(stream);
}	

std::string Lexer::peek()
{
	return _engine->peek();
}

std::string Lexer::location() const
{
	std::stringstream stream;
	
	stream << "(" << _engine->line << ":" << _engine->column << ")";
	
	return stream.str();
}

std::string Lexer::nextToken()
{
	auto result = _engine->nextToken();	

	hydrazine::log("Lexer") << "scanned token '" << result << "'\n";

	return result;
}

bool Lexer::scan(const std::string& token)
{
	hydrazine::log("Lexer") << "scanning for token '" << token << "'\n";
	
	return nextToken() == token;
}

void Lexer::scanThrow(const std::string& token)
{
	if(!scan(token))
	{
		throw std::runtime_error(location() + ": expecting a '" + token + "'");
	}
}

bool Lexer::scanPeek(const std::string& token)
{
	hydrazine::log("Lexer") << "scanning/peek for token '" << token << "'\n";
	
	return peek() == token;
}

void Lexer::reset()
{
	_engine->reset(_engine->stream);
}

void Lexer::checkpoint()
{
	_engine->checkpoint();
}

void Lexer::restoreCheckpoint()
{
	_engine->restore();
}

void Lexer::discardCheckpoint()
{
	assert(!_engine->checkpoints.empty());

	_engine->checkpoints.pop_back();
}

void Lexer::addTokenRegex(const std::string& regex)
{
	_engine->tokenRules.push_back(regex);
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

void LexerEngine::reset(std::istream* s)
{
	stream = s;
	
	stream->clear();
	stream->seekg(0, std::ios::beg);
	
	line   = 0;
	column = 0;
	
	checkpoints.clear();
}

void LexerEngine::checkpoint()
{
	checkpoints.push_back(_nextToken);
}

void LexerEngine::restore()
{
	assert(!checkpoints.empty());

	_nextToken = checkpoints.back();

	checkpoints.pop_back();
}

std::string LexerEngine::nextToken()
{
	auto result = peek();
	
	if(_nextToken != _tokens.end()) ++_nextToken;
	
	return result;
}

std::string LexerEngine::peek()
{
	if(_nextToken == _tokens.end()) return "";

	std::string result(_nextToken->endPosition -
		_nextToken->beginPosition, ' ');

	stream->seekg(_nextToken->beginPosition);
	
	stream->read((char*)result.data(), result.size());
	
	return result;
}

void LexerEngine::_createTokens()
{
	_tokens.clear();

	while(stream->good())
	{
		_tokens.push_back(TokenDescriptor(this));
	}
}

void LexerEngine::_mergeTokens()
{
	bool anyRemaining = !_tokens.empty();

	while(anyRemaining)
	{
		anyRemaining = false;
		
		LexerContextVector newTokens;
		
		// merge with neighbors
		for(auto token = _tokens.begin(); token != _tokens.end();)
		{
			if(token == _tokens.begin())
			{
				newTokens.push_back(_mergeWithBegin(*token));
				++token;
				continue;
			}
			
			auto next = token; ++next;
			
			if(next == _tokens.end())
			{
				newTokens.push_back(_mergeWithEnd(*token));
			}
			
			if(_canMerge(*token, *next))
			{
				newTokens.push_back(_mergeWithNext(*token, *next));
			}
			else
			{
				newTokens.push_back(*token);
				newTokens.push_back(*next );
			}
			
			token = ++next;
		}
		
		_tokens = std::move(newTokens);
	}
	
	_nextToken = _tokens.begin();
}

char LexerEngine::_snext()
{
	char c = stream->get();
	
	if(c == '\n')
	{
		++line;
		column = 0;
	}
	else
	{
		++column;
	}
	
	return c;
}

LexerEngine::TokenDescriptor LexerEngine::_mergeWithEnd(const TokenDescriptor& token);
LexerEngine::TokenDescriptor LexerEngine::_mergeWithNext(const TokenDescriptor& token,
	const TokenDescriptor& next);
	
LexerEngine::TokenDescriptor LexerEngine::_canMerge(const TokenDescriptor& token,
	const TokenDescriptor& next);


}

}


