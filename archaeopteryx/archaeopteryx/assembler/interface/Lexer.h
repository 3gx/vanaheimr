/*! \file   Lexer.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Monday September 12, 2011
	\brief  The header file for the Lexer class.
*/

#pragma once

/*! \brief A namespace for VIR assembler related classes and functions */
namespace assembler
{

/*! \brief An attempt at a fast data-parallel lexer for the VIR language 

	This intial attempt makes several language simplifications to ease the
	lexing process.  Specifically:
		1) Tokens may not cross new-line boundaries, including comments.
		2) Only one statement is allowed per line.

	Restriction 1) 

	To start with, we need a definition of the VIR language.
	
	1) Comments begin at ';' characters and continue to the end of the line.
	2) There are about 50 valid identifiers for instruction opcodes,
		modifiers, and types.
	
	
	Here is the general philosophy of the algorithm.
		1) The input is broken up into buckets,
			each is streamed through independently, producing an ordered
			bucket of tokens.
		2) 
*/
class Lexer
{
public:
	/*! \brief The constructor initializes itself from a file */
	Lexer(util::File* file);

};

}

