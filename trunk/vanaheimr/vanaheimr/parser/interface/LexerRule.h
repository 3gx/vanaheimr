/*! \file   LexerRule.h
	\date   April 28, 2013
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the LexerRule class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

namespace vanaheimr
{

namespace parser
{

/* \brief A class for representing a regular expression used to match a
		Lexer token
*/
class LexerRule
{
public:
	explicit LexerRule(const std::string& regex);
	
public:
	~LexerRule();
	LexerRule(const LexerRule&);
	
	LexerRule& operator=(const LexerRule& rule);
	
public:
	bool canMatchWithBegin(const std::string&) const;
	bool canMatchWithEnd(const std::string&) const;
	bool canOnlyMatchWithBegin(const std::string&) const;
	bool canOnlyMatchWithEnd(const std::string&) const;
	bool canMatch(const std::string&) const;
	bool isExactMatch(const std::string&) const;

public:
	const std::string& toString() const;
	
public:
	typedef std::string::iterator       iterator;
	typedef std::string::const_iterator const_iterator;
	
	typedef std::string::reverse_iterator       reverse_iterator;
	typedef std::string::const_reverse_iterator const_reverse_iterator;

public:
	      iterator begin();
	const_iterator begin() const;

	      iterator end();
	const_iterator end() const;
	
	      reverse_iterator rbegin();
	const_reverse_iterator rbegin() const;

	      reverse_iterator rend();
	const_reverse_iterator rend() const;	
	
public:
	bool   empty() const;
	size_t  size() const;

public:	
	class Character
	{
	public:
		virtual ~Character();
	
	public:
		virtual bool matches(const_iterator& position,
			const_iterator end) const = 0;
		virtual bool matches(const_reverse_iterator& position,
			const_reverse_iterator end) const;
	
	public:
		virtual Character* clone() const = 0;
	
	protected:
		typedef std::string::iterator       iterator;
		typedef std::string::const_iterator const_iterator;
	
		typedef std::string::reverse_iterator       reverse_iterator;
		typedef std::string::const_reverse_iterator const_reverse_iterator;	
	};
	
	typedef std::vector<Character*> CharacterVector;

private:
	typedef CharacterVector::iterator       regex_iterator;
	typedef CharacterVector::const_iterator const_regex_iterator;
	
	typedef CharacterVector::reverse_iterator reverse_regex_iterator;

	typedef CharacterVector::const_reverse_iterator
		const_reverse_regex_iterator;

private:
	void _interpretRegex(const std::string& regex);
	void _formRegex(const_iterator& begin, const_iterator end);

private:
	bool _match(const_iterator& matchEnd, const_regex_iterator& matchRuleEnd,
		const_iterator begin, const_iterator end,
		const_regex_iterator ruleBegin, const_regex_iterator ruleEnd) const;
	bool _match(const_iterator& matchEnd, const_iterator begin,
		const_iterator end) const;
	bool _isExactMatch(const std::string& text) const;
	bool _matchWithEnd(const_iterator begin, const_iterator end) const;
	bool _matchWithBegin(const_iterator begin, const_iterator end) const;
	bool _canMatchWithBegin(const std::string& text) const;
	bool _canMatchWithEnd(const std::string& text) const;
	bool _canMatch(const std::string&) const;
	
private:
	      regex_iterator regex_begin();
	const_regex_iterator regex_begin() const;

	      regex_iterator regex_end();
	const_regex_iterator regex_end() const;
	
	      reverse_regex_iterator regex_rbegin();
	const_reverse_regex_iterator regex_rbegin() const;

	      reverse_regex_iterator regex_rend();
	const_reverse_regex_iterator regex_rend() const;
	
private:
	CharacterVector _regex;
	std::string     _rawString;
	
};

}

}

