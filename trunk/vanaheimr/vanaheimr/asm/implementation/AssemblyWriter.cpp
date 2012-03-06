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

AssemblyWriter::AssemblyWriter()
{

}

void AssemblyWriter::write(std::ostream& stream, const ir::Module& module)
{
	report("Writing assembly for module '" << module.name << "'");

	for(auto function : module)
	{
		writeFunction(stream, function);
	}
	
	for(auto global = module.global_begin();
		global != module.global_end(); ++global)
	{
		writeGlobal(stream, *global);
	}
}

void AssemblyWriter::writeFunction(std::ostream& stream, const ir::Function& f)
{
	report(" For function '" << f.name << "'");
}

void AssemblyWriter::writeGlobal(std::ostream& stream, const ir::Global& g)
{
	report(" For global '" << g.name << "'");

}

}

}

