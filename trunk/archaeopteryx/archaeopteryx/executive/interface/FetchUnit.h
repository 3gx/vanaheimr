/*! \file   FetchUnit.h
	\date   Saturday April 23, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	        Sudnya  Diamos <mailsudnya@gmail.com>
	\brief  The header file for the FetchUnit class.
*/

#pragma once

namespace executive
{

/*! \brief The simulator's interface to a binary file */
class FetchUnit
{
public:
	/*! \brief Create a new fetch unit */
	__device__ FetchUnit(Binary* binary);
	__device__ InstructionContainer* getNextInstruction(PC);

};

}

