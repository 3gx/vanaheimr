/*! \file   LexerRule.h
	\date   April 28, 2013
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the LexerRule class.
*/

#pragma once

namespace vanaheimr
{

namespace parser
{

class LexerRule
{
public:
	explicit LexerRule(const std::string& regex);

public:
	const std::string& toString() const;

private:
	std::string _regex;

};

}

}

