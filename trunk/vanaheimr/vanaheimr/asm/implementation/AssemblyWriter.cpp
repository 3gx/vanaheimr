/*! \file   AssemblyWriter.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Sunday March 4, 2012
	\brief  The source file for the AssemblyWriter class.
*/

// Vanaheimr Includes
#include <vanaheimr/asm/interface/AssemblyWriter.h>

namespace vanaheimr
{

namespace asm
{

AssemblyWriter::AssemblyWriter();

void AssemblyWriter::write(std::ostream& stream, const ir::Module& m);

}

}

