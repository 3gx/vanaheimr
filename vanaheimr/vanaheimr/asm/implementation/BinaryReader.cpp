/*! 	\file   BinaryReader.cpp
	\date   Monday May 7, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the BinaryReader class.
*/

// Vanaheimr Includes
#include <vanaheimr/asm/interface/BinaryReader.h>

#include <vanaheimr/compiler/interface/Compiler.h>

#include <vanaheimr/ir/interface/Module.h>

namespace vanaheimr
{

namespace as
{

ir::Module* BinaryReader::read(std::istream& stream)
{
	_readHeader(stream);
	_readDataSection(stream);
	_readStringTable(stream);
	_readSymbolTable(stream);
	_readInstructions(stream);

	ir::Module* module = new ir::Module(_getName(),
		compiler::Compiler::getSingleton());
	
	return module;
}

}

}


