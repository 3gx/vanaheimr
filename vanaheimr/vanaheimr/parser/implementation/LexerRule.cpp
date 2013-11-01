/*! \file   LexerRule.cpp
	\date   April 28, 2013
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The source file for the LexerRule class.
*/

// Vanaheimr Includes
#include <vanaheimr/parser/interface/LexerRule.h>

// Standard Library Includes
#include <vector>
#include <cassert>
#include <algorithm>

namespace vanaheimr
{

namespace parser
{

LexerRule::LexerRule(const std::string& regex)
: _rawString(regex)
{
	_interpretRegex(regex);
}

LexerRule::~LexerRule()
{
	for(auto character : _regex)
	{
		delete character;
	}
}

LexerRule::LexerRule(const LexerRule& r)
: _rawString(r._rawString)
{
	_interpretRegex(_rawString);
}

LexerRule& LexerRule::operator=(const LexerRule& rule)
{
	if(&rule == this) return *this;
	
	for(auto character : _regex)
	{
		delete character;
	}
	
	_rawString = rule._rawString;
	
	_interpretRegex(_rawString);
	
	return *this;
}

bool LexerRule::canMatchWithBegin(const std::string& text) const
{
	return _matchWithBegin(text.begin(), text.end());
}

bool LexerRule::canMatchWithEnd(const std::string& text) const
{
	// poor man's backtracking, incorrect in some cases (TODO: real backtracking)
	for(auto beginPosition = text.begin();
		beginPosition != text.end(); ++beginPosition)
	{
		if(_matchWithEnd(beginPosition, text.end())) return true;
	}

	return false;
}

bool LexerRule::canOnlyMatchWithBegin(const std::string& text) const
{
	if(!canMatchWithBegin(text)) return false;
	
	return isExactMatch(text) || !canMatchWithEnd(text);
}

bool LexerRule::canOnlyMatchWithEnd(const std::string& text) const
{
	if(!canMatchWithEnd(text)) return false;
	
	return isExactMatch(text) || !canMatchWithBegin(text);
}
	
bool LexerRule::canMatch(const std::string& text) const
{
	for(auto beginPosition = text.begin();
		beginPosition != text.end(); ++beginPosition)
	{
		std::string::const_iterator position = beginPosition;
	
		if(_match(position, beginPosition, text.end())) return true;
	}

	return false;
}

bool LexerRule::isExactMatch(const std::string& text) const
{
	auto textMatchEnd = text.begin();
	auto ruleMatchEnd = regex_begin();
	
	if(!_match(textMatchEnd, ruleMatchEnd, text.begin(), text.end(),
		regex_begin(), regex_end()))
	{
		return false;
	}
	
	return textMatchEnd == text.end() && ruleMatchEnd == regex_end();
}

bool LexerRule::isEndRepeated() const
{
	if(empty()) return false;

	return (*regex_rbegin())->isRepeated();
}

const std::string& LexerRule::toString() const
{
	return _rawString;
}

LexerRule::iterator LexerRule::begin()
{
	return _rawString.begin();
}

LexerRule::const_iterator LexerRule::begin() const
{
	return _rawString.begin();
}

LexerRule::iterator LexerRule::end()
{
	return _rawString.end();
}

LexerRule::const_iterator LexerRule::end() const
{
	return _rawString.end();
}

LexerRule::reverse_iterator LexerRule::rbegin()
{
	return _rawString.rbegin();
}

LexerRule::const_reverse_iterator LexerRule::rbegin() const
{
	return _rawString.rbegin();
}

LexerRule::reverse_iterator LexerRule::rend()
{
	return _rawString.rend();
}

LexerRule::const_reverse_iterator LexerRule::rend() const
{
	return _rawString.rend();
}

bool LexerRule::empty() const
{
	return _rawString.empty();
}

size_t LexerRule::size() const
{
	return _rawString.size();
}

LexerRule::Character::~Character()
{

}

bool LexerRule::Character::matches(const_reverse_iterator& position,
	const_reverse_iterator end) const
{
	const_iterator forwardPosition     = --position.base();
	const_iterator nextForwardPosition = position.base();
	
	if(matches(forwardPosition, nextForwardPosition))
	{
		++position;
		return true;
	}
	
	return false;
}

bool LexerRule::Character::isRepeated() const
{
	return false;
}

void LexerRule::_interpretRegex(const std::string& regex)
{
	auto begin = regex.begin();
	auto end   = regex.end();
	
	while(begin != end)
	{
		_formRegex(begin, end);
	}
}

class NormalCharacter : public LexerRule::Character
{
public:
	explicit NormalCharacter(char c);
	
public:
	virtual bool matches(const_iterator& position,
		const_iterator end) const;

public:
	virtual Character* clone() const;

public:
	char character;
};

NormalCharacter::NormalCharacter(char c)
: character(c)
{

}

bool NormalCharacter::matches(const_iterator& position,
	const_iterator end) const
{
	if(character == *position)
	{
		++position;
		return true;
	}
	
	return false;
}

LexerRule::Character* NormalCharacter::clone() const
{
	return new NormalCharacter(*this);
}

class AnyCharacter : public LexerRule::Character
{	
public:
	virtual bool matches(const_iterator& position,
		const_iterator end) const;

public:
	virtual Character* clone() const;
};

bool AnyCharacter::matches(const_iterator& position, const_iterator end) const
{
	char character = *position;
	
	if(character != '\n')
	{
		++position;

		return true;
	}
	
	return false;
}

LexerRule::Character* AnyCharacter::clone() const
{
	return new AnyCharacter;
}

/* \brief Matches zero or more repeated instances */
class RepeatedCharacter : public LexerRule::Character
{	
public:
	explicit RepeatedCharacter(Character* repeated);

public:
	~RepeatedCharacter();
	RepeatedCharacter(const RepeatedCharacter&);
	RepeatedCharacter& operator=(const RepeatedCharacter&);

public:
	virtual bool matches(const_iterator& position,
		const_iterator end) const;
	virtual bool matches(const_reverse_iterator& position,
			const_reverse_iterator end) const;
public:
	virtual bool isRepeated() const;

public:
	virtual Character* clone() const;

private:
	Character* _subCharacter;

};

RepeatedCharacter::RepeatedCharacter(Character* repeated)
: _subCharacter(repeated)
{

}

RepeatedCharacter::~RepeatedCharacter()
{
	delete _subCharacter;
}

RepeatedCharacter::RepeatedCharacter(const RepeatedCharacter& r)
: _subCharacter(r._subCharacter->clone())
{

}

RepeatedCharacter& RepeatedCharacter::operator=(const RepeatedCharacter& r)
{
	if(&r == this) return *this;

	auto temporaryCharacter = r._subCharacter->clone();

	delete _subCharacter;

	_subCharacter = temporaryCharacter;
	
	return *this;
}

bool RepeatedCharacter::matches(const_iterator& position,
	const_iterator end) const
{
	while(position != end)
	{
		if(!_subCharacter->matches(position, end))
		{
			break;
		}
	}
	
	return true;
}

bool RepeatedCharacter::matches(const_reverse_iterator& position,
	const_reverse_iterator end) const
{
	while(position != end)
	{
		if(!_subCharacter->matches(position, end))
		{
			break;
		}
	}
	
	return true;
}

bool RepeatedCharacter::isRepeated() const
{
	return true;
}

LexerRule::Character* RepeatedCharacter::clone() const
{
	return new RepeatedCharacter(*this);
}

class NumericCharacter : public LexerRule::Character
{
public:
	virtual bool matches(const_iterator& position,
		const_iterator end) const;

public:
	virtual Character* clone() const;
};

static bool isNumeric(char c)
{
	return (c >= '0') && (c <= '9');
}

static bool isLowerCaseAlpha(char c)
{
	return (c >= 'a') && (c <= 'z');
}

static bool isUpperCaseAlpha(char c)
{
	return (c >= 'A') && (c <= 'Z');
}

bool NumericCharacter::matches(const_iterator& position,
	const_iterator end) const
{
	auto character = *position;
	
	if(isNumeric(character))
	{
		 ++position;
		 return true;
	}
	
	return false;
}

LexerRule::Character* NumericCharacter::clone() const
{
	return new NumericCharacter;
}

class AlphaNumericCharacter : public LexerRule::Character
{
public:
	virtual bool matches(const_iterator& position,
		const_iterator end) const;

public:
	virtual Character* clone() const;
};

bool AlphaNumericCharacter::matches(const_iterator& position,
	const_iterator end) const
{
	auto character = *position;
	
	bool result = isLowerCaseAlpha(character) || isUpperCaseAlpha(character) || 
		isNumeric(character);

	if(result)
	{
		++position;
		return true;
	}
	
	return false;
}

LexerRule::Character* AlphaNumericCharacter::clone() const
{
	return new AlphaNumericCharacter;
}

class CharacterRange : public LexerRule::Character
{
public:
	CharacterRange(char begin, char end);

public:
	virtual bool matches(const_iterator& position,
		const_iterator end) const;
	
public:
	virtual Character* clone() const;
	
private:
	char _begin;
	char _end;
};

CharacterRange::CharacterRange(char begin, char end)
: _begin(begin), _end(end)
{

}

bool CharacterRange::matches(const_iterator& position,
	const_iterator end) const
{
	char character = *position;
	
	if(_begin <= character && character <= _end)
	{
		++position;
		return true;
	}
	
	return false;
}

LexerRule::Character* CharacterRange::clone() const
{
	return new CharacterRange(_begin, _end);
}

typedef LexerRule::CharacterVector CharacterVector;

class NewCharacterClass : public LexerRule::Character
{
public:
	NewCharacterClass(const CharacterVector& characters, bool invert);

public:
	~NewCharacterClass();
	NewCharacterClass(const NewCharacterClass&);
	NewCharacterClass& operator=(const NewCharacterClass&);

public:
	virtual bool matches(const_iterator& position,
		const_iterator end) const;
	
public:
	virtual Character* clone() const;
	
private:
	CharacterVector _classMembers;
	bool            _invert;
	
};

NewCharacterClass::NewCharacterClass(const CharacterVector& m, bool i)
: _classMembers(m), _invert(i)
{

}

NewCharacterClass::~NewCharacterClass()
{
	for(auto c : _classMembers)
	{
		delete c;
	}
}

NewCharacterClass::NewCharacterClass(const NewCharacterClass& c)
: _invert(c._invert)
{
	for(auto classMembers : c._classMembers)
	{
		_classMembers.push_back(classMembers->clone());
	}
}

NewCharacterClass& NewCharacterClass::operator=(const NewCharacterClass& c)
{
	if(&c == this) return *this;
	
	for(auto c : _classMembers)
	{
		delete c;
	}

	_classMembers.clear();
	
	for(auto classMembers : c._classMembers)
	{
		_classMembers.push_back(classMembers->clone());
	}
	
	_invert = c._invert;
	
	return *this;
}

bool NewCharacterClass::matches(const_iterator& position,
	const_iterator end) const
{
	bool result = false;
	
	for(auto characterClass : _classMembers)
	{
		if(characterClass->matches(position, end))
		{
			result = true;
			break;
		}
	}
	
	bool finalResult = result ^ _invert;

	if(finalResult)
	{
		if(_invert)
		{
			++position;
		}
	}
	else
	{
		if(_invert)
		{
			--position;
		}
	}

	return finalResult;
}

LexerRule::Character* NewCharacterClass::clone() const
{
	return new NewCharacterClass(*this);
}

static bool containsString(std::string::const_iterator begin, 
	std::string::const_iterator end, const std::string& string)
{
	return std::string(begin, end).find(string) == 0;
}

static bool isCharacterClass(std::string::const_iterator begin,
	std::string::const_iterator end)
{
	if(begin == end)
	{
		return false;
	}

	if(*begin != '[')
	{
		return false;
	}

	auto position = std::find(begin, end, ']');

	return position != end;
}

static bool isRange(std::string::const_iterator begin, 
	std::string::const_iterator end)
{
	if(std::distance(begin, end) < 3)
	{
		return false;
	}
	
	if(begin[1] != '-')
	{
		return false;
	}
	
	return true;
}

static LexerRule::Character* parseCharacterClass(
	std::string::const_iterator& begin, 
	std::string::const_iterator end)
{
	// skip the [
	++begin;

	// find the ]
	auto endOfClass = std::find(begin, end, ']');

	assert(endOfClass != end);	

	bool invert = false;

	if(*begin == '^')
	{
		++begin;
		invert = true;
	}
	
	CharacterVector members;
	
	while(begin != endOfClass)
	{
		if(isRange(begin, endOfClass))
		{
			char rangeBegin = *begin;
			char rangeEnd   = begin[2];
			
			members.push_back(new CharacterRange(rangeBegin, rangeEnd));
			
			begin += 3;
		}
		else
		{
			char character = *begin; ++begin;
			
			members.push_back(new NormalCharacter(character));
		}
	}
	
	begin = endOfClass + 1;
	
	return new NewCharacterClass(members, invert);
}

void LexerRule::_formRegex(const_iterator& begin, const_iterator end)
{
	if(*begin == '\\')
	{
		// Handle an escape
		++begin;
		
		assert(begin != end);
		
		// Handle a normal character
		char character = *begin; ++begin;
	
		_regex.push_back(new NormalCharacter(character));

	}
	else if(*begin == '.')
	{
		// Handle a wildcard
		++begin;
		
		_regex.push_back(new AnyCharacter());
	}
	else if(containsString(begin, end, "[:alnum:]"))
	{
		begin += sizeof("[:alnum:]");
		
		_regex.push_back(new AlphaNumericCharacter());
	}
	else if(containsString(begin, end, "[:digit:]"))
	{
		begin += sizeof("[:digit:]");
		
		_regex.push_back(new NumericCharacter());
	}
	else if(isCharacterClass(begin, end))
	{
		std::string classMembers;

		Character* newCharacter = parseCharacterClass(begin, end);
		
		_regex.push_back(newCharacter);
	}
	else
	{
		// Handle a normal character
		char character = *begin; ++begin;
	
		_regex.push_back(new NormalCharacter(character));
	}
	
	if(begin == end)
	{
		return;
	}
	
	if(*begin == '*')
	{
		// Repeat the last character
		++begin;
		
		_regex.back() = new RepeatedCharacter(_regex.back());
	}
}

bool LexerRule::_match(const_iterator& matchEnd,
	const_regex_iterator& matchRuleEnd,
	const_iterator begin, const_iterator end,
	const_regex_iterator ruleBegin, const_regex_iterator ruleEnd) const
{
	auto ruleCharacter = ruleBegin;
	auto originalBegin = begin;
	
	for( ; ruleCharacter != ruleEnd; )
	{
		if(!(*ruleCharacter)->matches(begin, end))
		{
			return false;
		}
		
		// This is a match
		++ruleCharacter;
		
		if(begin == end) break;
	}
	
	matchEnd     = begin;
	matchRuleEnd = ruleCharacter;
	
	return originalBegin != begin;
}

bool LexerRule::_match(const_iterator& matchEnd,
	const_iterator textBegin, const_iterator textEnd) const
{
	auto ruleEnd = regex_begin();
	
	for(auto ruleCharacter = regex_begin(); ruleCharacter != regex_end();
		++ruleCharacter)
	{
		if(_match(matchEnd, ruleEnd, textBegin, textEnd, ruleCharacter,
			regex_end()))
		{
			return true;
		}
	}
	
	return false;
}

bool LexerRule::_matchWithEnd(const_iterator begin, const_iterator end) const
{
	std::string::const_reverse_iterator textRbegin(end);
	std::string::const_reverse_iterator textRend(begin);
	
	for(auto ruleCharacter = regex_rbegin(); ruleCharacter != regex_rend(); )
	{
		if(!(*ruleCharacter)->matches(textRbegin, textRend))
		{
			return false;
		}
		
		++ruleCharacter;
		
		if(textRbegin == textRend) break;
	}
	
	return true;
}

bool LexerRule::_matchWithBegin(const_iterator textBegin,
	const_iterator textEnd) const
{
	for(auto ruleCharacter = regex_begin(); ruleCharacter != regex_end();
		++ruleCharacter)
	{
		if(!(*ruleCharacter)->matches(textBegin, textEnd))
		{
			return false;
		}

		if(textBegin == textEnd) break;
	}
	
	return true;
}

LexerRule::regex_iterator LexerRule::regex_begin()
{
	return _regex.begin();
}

LexerRule::const_regex_iterator LexerRule::regex_begin() const
{
	return _regex.begin();
}

LexerRule::regex_iterator LexerRule::regex_end()
{
	return _regex.end();
}

LexerRule::const_regex_iterator LexerRule::regex_end() const
{
	return _regex.end();
}

LexerRule::reverse_regex_iterator LexerRule::regex_rbegin()
{
	return _regex.rbegin();
}

LexerRule::const_reverse_regex_iterator LexerRule::regex_rbegin() const
{
	return _regex.rbegin();
}

LexerRule::reverse_regex_iterator LexerRule::regex_rend()
{
	return _regex.rend();
}

LexerRule::const_reverse_regex_iterator LexerRule::regex_rend() const
{
	return _regex.rend();
}

}

}

