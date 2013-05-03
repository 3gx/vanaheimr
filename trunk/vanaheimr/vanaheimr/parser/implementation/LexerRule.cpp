/*! \file   LexerRule.cpp
	\date   April 28, 2013
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The source file for the LexerRule class.
*/

// Vanaheimr Includes
#include <vanaheimr/parser/interface/LexerRule.h>

namespace vanaheimr
{

namespace parser
{

LexerRule::LexerRule(const std::string& regex)
: _regex(regex)
{
	// record wildcards
}

bool LexerRule::canMatchWithBegin(const std::string& text) const
{
	return _matchWithBegin(text.begin(), text.end());
}

bool LexerRule::canMatchWithEnd(const std::string& text) const
{
	for(auto beginPosition = text.begin();
		beginPosition != text.end(); ++beginPosition)
	{
		if(_matchWithEnd(beginPosition, text.end())) return true;
	}
	
	return false;
}

bool LexerRule::canMatch(const std::string&) const
{
	for(auto beginPosition = text.begin();
		beginPosition != text.end(); ++beginPosition)
	{
		std::string::const_iterator position = beginPosition;
	
		if(_match(position, beginPosition, text.end())) return true;
	}

	return false;
}

const std::string& LexerRule::toString() const
{
	return _regex;
}

LexerRule::iterator LexerRule::begin();
LexerRule::const_iterator LexerRule::begin() const;

LexerRule::iterator LexerRule::end();
LexerRule::const_iterator LexerRule::end() const;

bool LexerRule::isExactMatch(const std::string& text) const
{
	auto textMatchEnd = text.begin();
	auto ruleMatchEnd = rule.begin();
	
	if(!_match(textMatchEnd, ruleMatchEnd, text.begin(), text.end(),
		rule.begin(), rule.end()))
	{
		return false;
	}
	
	return textMatchEnd == text.end() && ruleMatchEnd == rule.end();
}

bool LexerRule::_match(const_iterator& matchEnd,
	const_iterator& matchRuleEnd,
	const_iterator begin, const_iterator end,
	const_iterator ruleBegin, const_iterator ruleEnd) const
{
	auto ruleCharacter = ruleBegin;
	
	for( ; ruleCharacter != ruleEnd; )
	{
		auto ruleNextCharacter = ruleCharacter; ++ruleNextCharacter;

		if(_isWildcard(ruleCharacter))
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

bool LexerRule::_match(const_iterator& matchEnd,
	const_iterator begin, const_iterator end) const
{
	auto ruleEnd = begin();
	
	for(auto ruleCharacter = begin(); ruleCharacter != end(); ++ruleCharacter)
	{
		if(match(matchEnd, ruleEnd, begin, end, ruleCharacter, rule.end()))
		{
			return true;
		}
	}
	
	return false;
}

bool LexerRule::_matchWithEnd(const_iterator begin, const_iterator end) const
{
	std::string::const_reverse_iterator rbegin(end);
	std::string::const_reverse_iterator rend(begin);
	
	for(auto ruleCharacter = rbegin(); ruleCharacter != rend(); )
	{
		auto ruleNextCharacter = ruleCharacter; ++ruleNextCharacter;

		if(isWildcard(ruleCharacter))
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

bool LexerRule::_matchWithBegin(const_iterator begin, const_iterator end) const
{
	for(auto ruleCharacter = begin(); ruleCharacter != end(); )
	{
		auto ruleNextCharacter = ruleCharacter; ++ruleNextCharacter;

		if(isWildcard(ruleCharacter))
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

}

}

