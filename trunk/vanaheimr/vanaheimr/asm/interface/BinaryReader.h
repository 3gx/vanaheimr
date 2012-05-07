/*! 	\file   BinaryReader.h
	\date   Monday May 7, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the BinaryReader class.
*/

#pragma once

// Archaeopteryx Includes
#include <archaeopteryx/ir/interface/Instruction.h>

namespace vanaheimr
{

namespace as
{

/*! \brief Reads in a vanaheimr bytecode file yielding a module. */
class BinaryReader
{
public:
	/*! \brief Attempts to read from a binary stream, returns a module */
	ir::Module* read(std::istream& stream);



};

}

}


