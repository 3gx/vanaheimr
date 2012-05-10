/*! 	\file   BinaryReader.cpp
	\date   Monday May 7, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the BinaryReader class.
*/

// Vanaheimr Includes
#include <vanaheimr/asm/interface/BinaryReader.h>

namespace vanaheimr
{

namespace as
{

ir::Module* BinaryReader::read(std::istream& stream)
{
	_readHeader(stream);
	_read
}

}

}


