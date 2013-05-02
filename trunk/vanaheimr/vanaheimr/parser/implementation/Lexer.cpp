/*! \file   Lexer.cpp
	\date   April 9, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the Lexer class.
*/

// Vanaheimr Includes
#include <vanaheimr/parser/interface/Lexer.h>
#include <vanaheimr/parser/interface/LexerRule.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <vector>
#include <sstream>
#include <cassert>
#include <stdexcept>
#include <set>
#include <algorithm>

namespace vanaheimr
{

namespace parser
{

class LexerEngine
{
public:
	// TODO: Consider a rule representation that uses a state machine
	// rather than an explicit set of rules that could be matched
	typedef std::set<LexerRule*> RuleSet;

	class TokenDescriptor
	{
	public:
		explicit TokenDescriptor(LexerEngine* engine);
		TokenDescriptor(const TokenDescriptor& left,
			const TokenDescriptor& right);
	
	public:
		size_t beginPosition;
		size_t endPosition;
	
	public:
		size_t line;
		size_t column;
		
	public:
		RuleSet possibleMatches;
	
	public:
		LexerEngine* engine;
		
	public:
		bool   isEndMatched()   const;
		bool   isBeginMatched() const;
		bool   isMatched()      const;
		size_t size()           const;
		
	public:
		std::string getString() const;
	};
	
	typedef std::vector<TokenDescriptor> TokenVector;
	typedef TokenVector::iterator LexerContext;

	typedef std::vector<LexerContext> LexerContextVector;

	typedef std::vector<LexerRule> RuleVector;

public:
	std::istream* stream;

	size_t line;
	size_t column;

	LexerContextVector checkpoints;

public:
	RuleVector tokenRules;
	RuleVector whitespaceRules;

public:
	std::string nextToken();
	std::string peek();
	bool hitEndOfStream() const;

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
	void _filterWithNeighbors(const LexerContext& token);

	TokenDescriptor _mergeWithEnd(const LexerContext& token);
	TokenDescriptor _mergeWithNext(const LexerContext& token,
		const LexerContext& next);
	
	bool _isAMergePossible(const LexerContext& token,
		const LexerContext& next);
	bool _canMerge(const LexerContext& token,
		const LexerContext& next);
	
	bool _isNewToken(const LexerContext& token);
	bool _couldBeTokenEnd(const LexerContext& token);
	bool _couldBeTokenBegin(const LexerContext& token);
	
	bool _canMatch(const std::string& rule,
		const std::string& text);
	
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

bool Lexer::hitEndOfStream() const
{
	return _engine->hitEndOfStream();
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
	_engine->tokenRules.push_back(LexerRule(regex));
}

void Lexer::addWhitespaceRules(const std::string& whitespaceCharacters)
{
	for(auto& character : whitespaceCharacters)
	{
		_engine->whitespaceRules.push_back(
			LexerRule(std::string(1, character)));
	}
}

void Lexer::addTokens(const StringList& regexes)
{
	for(auto& regex : regexes)
	{
		addTokenRegex(regex);
	}
}

void LexerEngine::reset(std::istream* s)
{
	stream = s;
	
	stream->clear();
	stream->seekg(0, std::ios::beg);
	
	line   = 0;
	column = 0;
	
	checkpoints.clear();
	
	// Create the entire set of tokens
	_createTokens();
	_mergeTokens();	
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
	if(hitEndOfStream()) return "";

	std::string result(_nextToken->endPosition -
		_nextToken->beginPosition, ' ');

	stream->seekg(_nextToken->beginPosition);
	
	stream->read((char*)result.data(), result.size());
	
	return result;
}

bool LexerEngine::hitEndOfStream() const
{
	return _nextToken == _tokens.end();
}

void LexerEngine::_createTokens()
{
	_tokens.clear();

	hydrazine::log("Lexer") << "Creating initial tokens...\n";

	stream->seekg(0, std::ios::end);

	size_t streamSize = stream->tellg();

	stream->seekg(0, std::ios::beg);

	for(size_t i = 0; i < streamSize; ++i)
	{
		_tokens.push_back(TokenDescriptor(this));
			
		_snext();
	}
}

void LexerEngine::_mergeTokens()
{
	hydrazine::log("Lexer") << "Merging partial tokens together...\n";

	unsigned int counter = 0;
	unsigned int unmatchedTokenCount = _tokens.size();
	
	while(true)
	{
		hydrazine::log("Lexer") << "============== Iteration "
			<< counter++ << " ==============\n";

		hydrazine::log("Lexer") << " Filtering out matched tokens:\n";
		
		// Filter out matched tokens
		// Parallel for-all, start from previous unmatched count
		unsigned int unmatchedCount = 0;
		for(auto token = _tokens.begin(); token != _tokens.end(); ++token)
		{
			if(token->isMatched()) continue;
			
			_filterWithNeighbors(token);
			
			if(token->isMatched())
			{
				hydrazine::log("Lexer") << "  Token '" << token->getString()
					<< "' matched rule '" << (*token->possibleMatches.begin())
					<< "'\n";
		
				continue;
			}
			
			++unmatchedCount;
		}

		hydrazine::log("Lexer") << " unmatched token count "
			<< unmatchedCount << "\n";
		
		if(unmatchedCount == 0) break;
		
		assertM(counter < 2 || unmatchedCount < unmatchedTokenCount,
			"Lexing did not make forward progress during this iteration.");

		unmatchedTokenCount = unmatchedCount;
		
		TokenVector newTokens;
	
		hydrazine::log("Lexer") << " Merging unmatched tokens with neighbors\n";
		
		// merge with neighbors
		// Parallel for-all
		for(auto token = _tokens.begin(); token != _tokens.end(); ++token)
		{
			if(token->isMatched())
			{
				newTokens.push_back(*token);
				continue;
			}
		
			auto next = token; ++next;

			hydrazine::log("Lexer") << "  For unmatched token '"
				<< token->getString() << "'\n";
			
			if(next == _tokens.end())
			{
				hydrazine::log("Lexer") << "   attempting to merge with "
					"end of stream.\n";
				newTokens.push_back(_mergeWithEnd(token));
				continue;
			}
			else
			{
				hydrazine::log("Lexer") << "   attempting to merge with '" <<
					next->getString() << "'\n";
				
				if(_canMerge(token, next))
				{
					hydrazine::log("Lexer") << "    success\n";
					newTokens.push_back(_mergeWithNext(token, next));
				}
				else
				{
					hydrazine::log("Lexer") << "    failed\n";
					newTokens.push_back(*token);
					newTokens.push_back(*next );
				}
			}
			
			++token;
		}
		
		// Update tokens
		// Parallel stream compaction
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

static bool isWildcard(char c)
{
	return c == '*';
}

static bool match(std::string::const_iterator& matchEnd,
	std::string::const_iterator& matchRuleEnd,
	std::string::const_iterator begin, std::string::const_iterator end,
	std::string::const_iterator ruleBegin, std::string::const_iterator ruleEnd)
{
	auto ruleCharacter = ruleBegin;
	for( ; ruleCharacter != ruleEnd; )
	{
		auto ruleNextCharacter = ruleCharacter; ++ruleNextCharacter;

		if(isWildcard(*ruleCharacter))
		{
			if(ruleNextCharacter != ruleEnd)
			{
				if(*ruleNextCharacter == *begin)
				{
					ruleCharacter = ruleNextCharacter;
					++ruleCharacter;
				}
			}
			
			++begin;
			
			if(begin == end) break;
			
			continue;
		}
		
		if(*ruleCharacter != *begin)
		{
			return false;
		}
		
		++ruleCharacter;
		++begin;
		
		if(begin == end) break;
	}
	
	matchEnd     = begin;
	matchRuleEnd = ruleCharacter;
	
	return true;
}

static bool match(std::string::const_iterator& matchEnd,
	std::string::const_iterator begin,
	std::string::const_iterator end, const std::string& rule)
{
	auto ruleEnd = rule.begin();
	
	for(auto ruleCharacter = rule.begin(); ruleCharacter != rule.end();
		++ruleCharacter)
	{
		if(match(matchEnd, ruleEnd, begin, end, ruleCharacter, rule.end()))
		{
			return true;
		}
	}
	
	return false;
}

static bool exactMatch(const std::string& text, const std::string& rule)
{
	auto textMatchEnd = text.begin();
	auto ruleMatchEnd = rule.begin();
	
	if(!match(textMatchEnd, ruleMatchEnd, text.begin(), text.end(),
		rule.begin(), rule.end()))
	{
		return false;
	}
	
	return textMatchEnd == text.end() && ruleMatchEnd == rule.end();
}

static bool matchWithEnd(std::string::const_iterator begin,
	std::string::const_iterator end, const std::string& rule)
{
	std::string::const_reverse_iterator rbegin(end);
	std::string::const_reverse_iterator rend(begin);
	
	for(auto ruleCharacter = rule.rbegin(); ruleCharacter != rule.rend(); )
	{
		auto ruleNextCharacter = ruleCharacter; ++ruleNextCharacter;

		if(isWildcard(*ruleCharacter))
		{
			if(ruleNextCharacter != rule.rend())
			{
				if(*ruleNextCharacter == *rbegin)
				{
					++ruleCharacter;
				}
			}
			
			++rbegin;
			if(rbegin == rend) break;
			continue;
		}
		
		if(*ruleCharacter != *rbegin)
		{
			return false;
		}
		
		++ruleCharacter;
		++rbegin;
		
		if(rbegin == rend) break;
	}
	
	return true;
}

static bool matchWithBegin(std::string::const_iterator begin,
	std::string::const_iterator end, const std::string& rule)
{
	for(auto ruleCharacter = rule.begin(); ruleCharacter != rule.end(); )
	{
		auto ruleNextCharacter = ruleCharacter; ++ruleNextCharacter;

		if(isWildcard(*ruleCharacter))
		{
			if(ruleNextCharacter != rule.end())
			{
				if(*ruleNextCharacter == *begin)
				{
					++ruleCharacter;
				}
			}
			
			++begin;
			if(begin == end) break;
			continue;
		}
		
		if(*ruleCharacter != *begin)
		{
			return false;
		}
		
		++ruleCharacter;
		++begin;
		
		if(begin == end) break;
	}
	
	return true;
}

static bool canMatchWithEnd(const std::string& rule, const std::string& text)
{
	assert(!rule.empty());
	assert(!text.empty());

	for(auto beginPosition = text.begin();
		beginPosition != text.end(); ++beginPosition)
	{
		if(matchWithEnd(beginPosition, text.end(), rule)) return true;
	}
	
	return false;
}

static bool canMatchWithBegin(const std::string& rule, const std::string& text)
{
	assert(!rule.empty());
	assert(!text.empty());

	return matchWithBegin(text.begin(), text.end(), rule);
}

void LexerEngine::_filterWithNeighbors(const LexerContext& token)
{
	hydrazine::log("Lexer") << "  checking token possible matches for '" <<
		token->getString() << "'\n";
	
	bool isNewToken = _isNewToken(token);

	auto next = token; ++next;
	
	bool isTokenEnd = next == _tokens.end();
	
	if(!isTokenEnd)
	{
		isTokenEnd = !_isAMergePossible(token, next);
	}

	hydrazine::log("Lexer") << "   possible matches for '" <<
		token->getString() << "'"
		<< (isNewToken ? " (starts new token)":"") 
		<< (isTokenEnd ? " (ends current token)":"") << "\n";

	LexerEngine::RuleSet remainingRules;

	auto tokenString = token->getString();

	for(auto rule : token->possibleMatches)
	{
		if(isNewToken && !rule->canMatchWithBegin(tokenString)) continue;
		if(isTokenEnd &&   !rule->canMatchWithEnd(tokenString)) continue;
		if(!rule->canMatch(tokenString)) continue;
		
		hydrazine::log("Lexer") << "    '" << rule->toString() << "'\n";
	
		remainingRules.insert(rule);
	}
	
	assertM(!remainingRules.empty(), "No possible matched for token '"
		<< token->getString() << "'");
	
	token->possibleMatches = std::move(remainingRules);
}

LexerEngine::TokenDescriptor LexerEngine::_mergeWithEnd(
	const LexerContext& token)
{
	auto string = token->getString();
	
	TokenDescriptor newToken(*token);

	hydrazine::log("Lexer") << "   possible matches:\n";
	
	for(auto rule : token->possibleMatches)
	{
		if(rule->canMatchWithEnd(string))
		{
			hydrazine::log("Lexer") << "    '" << rule->toString() << "'\n";
			newToken.possibleMatches.insert(rule);
		}
	}
	
	return newToken;
}

static LexerEngine::RuleSet intersection(const LexerEngine::RuleSet& left,
	const LexerEngine::RuleSet& right)
{
	LexerEngine::RuleSet result;
	
	std::set_intersection(left.begin(), left.end(), right.begin(), right.end(),
		std::inserter(result, result.end()));
	
	return result;
}

LexerEngine::TokenDescriptor LexerEngine::_mergeWithNext(
	const LexerContext& token,
	const LexerContext& next)
{
	hydrazine::log("Lexer") << "   merging '" << token->getString()
		<< "' with '" << next->getString() << "':\n";
	
	TokenDescriptor newToken(*token, *next);
	
	auto string = newToken.getString();
	
	// The set of possible matches is the intersection of the two tokens
	auto possibleMatches = intersection(token->possibleMatches,
		next->possibleMatches);
	
	// Only keep matches that handle the combined string
	hydrazine::log("Lexer") << "    possible rule matches:\n";
	for(auto rule : possibleMatches)
	{
		if(_rule->canMatch(string))
		{
			hydrazine::log("Lexer") << "     '" << rule->toString() << "'\n";
			newToken.possibleMatches.insert(rule);
		}
	}
	
	return newToken;
}

bool LexerEngine::_isAMergePossible(
	const LexerContext& token,
	const LexerContext& next)
{
	auto mergedToken = _mergeWithNext(token, next);

	return !mergedToken.possibleMatches.empty();
}
	
bool LexerEngine::_canMerge(
	const LexerContext& token,
	const LexerContext& next)
{
	// Can merge if there is no ambiguity about the rule matched
	//  and both tokens match the same rule
	if(next->possibleMatches.size() == 1)
	{
		if(next->couldBeTokenBegin())
		{
			return true;
		}
	}
	
	// Can't match if there is ambiguity about the left being a token end
	if(token->couldBeTokenEnd())
	{
		if(!token->isNewToken())
		{
			hydrazine::log("Lexer") << "     can't merge, "
				"left could be a token end.\n";
			return false;
		}
	}
	
	// Or the right being a token begin
	if(!token->isBeginMatched())
	{
		if(next->couldBeTokenBegin())
		{
			hydrazine::log("Lexer") << "     can't merge, "
				"right could be a token begin.\n";
	
			return false;
		}
	}
	
	return true;
}

bool LexerEngine::_isNewToken(const LexerContext& token)
{
	bool isNewToken = token == _tokens.begin();

	if(!isNewToken)
	{
		auto previous = token; --previous;
		
		isNewToken = previous->isEndMatched();
	}
	
	return isNewToken;
}	

bool LexerEngine::_couldBeTokenEnd(const LexerContext& token)
{
	auto string = token->getString();
	
	for(auto possibleMatch : token->possibleMatches)
	{
		if(canMatchWithEnd(*possibleMatch, string)) return true;
	}
	
	return false;
}

bool LexerEngine::_couldBeTokenBegin(const LexerContext& token)
{
	auto string = token->getString();
	
	for(auto possibleMatch : token->possibleMatches)
	{
		if(canMatchWithBegin(*possibleMatch, string)) return true;
	}
	
	return false;
}

bool LexerEngine::_canMatch(const std::string& rule,
	const std::string& text)
{
	assert(!rule.empty());
	assert(!text.empty());

	for(auto beginPosition = text.begin();
		beginPosition != text.end(); ++beginPosition)
	{
		std::string::const_iterator position = beginPosition;
	
		if(match(position, beginPosition, text.end(), rule)) return true;
	}

	return false;
}


LexerEngine::TokenDescriptor::TokenDescriptor(LexerEngine* e)
: beginPosition(e->stream->tellg()),
  endPosition((size_t)e->stream->tellg() + 1),
  line(e->line), column(e->column), engine(e)
{
	hydrazine::log("Lexer") << " created new token '"
		<< getString() << "'\n";

	for(auto& rule : engine->tokenRules)
	{
		hydrazine::log("Lexer") << "  could match rule '" << rule << "'\n";
		possibleMatches.insert(&rule);
	}
	
	for(auto& rule : engine->whitespaceRules)
	{
		hydrazine::log("Lexer") << "  could match rule '" << rule << "'\n";
		possibleMatches.insert(&rule);
	}
	
	if(isMatched())
	{
		hydrazine::log("Lexer") << "  Token '" << getString()
			<< "' matched rule '" << **possibleMatches.begin()
			<< "'\n";
	}
}

LexerEngine::TokenDescriptor::TokenDescriptor(const TokenDescriptor& left,
	const TokenDescriptor& right)
: beginPosition(left.beginPosition),
  endPosition(right.endPosition),
  line(left.line), column(right.column), engine(left.engine)
{
	assert(left.endPosition == right.beginPosition);
}

bool LexerEngine::TokenDescriptor::isBeginMatched() const
{
	assertM(!possibleMatches.empty(), "No possible matched for token '"
		<< getString() << "'");
	
	if(possibleMatches.size() > 1) return false;

	return canMatchWithBegin(**possibleMatches.begin(), getString()) &&
		!canMatchWithEnd(**possibleMatches.begin(), getString());
}

bool LexerEngine::TokenDescriptor::isEndMatched() const
{
	assert(!possibleMatches.empty());
	
	if(possibleMatches.size() > 1) return false;

	return canMatchWithEnd(**possibleMatches.begin(), getString()) &&
		!canMatchWithBegin(**possibleMatches.begin(), getString());
}

bool LexerEngine::TokenDescriptor::isMatched() const
{
	if(possibleMatches.size() > 1) return false;

	return exactMatch(getString(), **possibleMatches.begin());	
}

size_t LexerEngine::TokenDescriptor::size() const
{
	return endPosition - beginPosition;
}

std::string LexerEngine::TokenDescriptor::getString() const
{
	auto mutableEngine = const_cast<LexerEngine*>(engine);

	std::string result(size(), ' ');

	auto position = mutableEngine->stream->tellg();

	mutableEngine->stream->seekg(beginPosition, std::ios::beg);
	
	mutableEngine->stream->read((char*)(result.data()), size());

	mutableEngine->stream->seekg(position, std::ios::beg);
	
	return result;
}

}

}


