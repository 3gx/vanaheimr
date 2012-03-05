/*! \file   AssemblyWriter.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Sunday March 4, 2012
	\brief  The header file for the AssemblyWriter class.
*/

#pragma once

// Forward Declarations
namespace vanaheimr{ namespace ir { class Module; } }

namespace vanaheimr
{

namespace asm
{

/*! \brief Used to write a module to an object file */
class AssemblyWriter
{
public:
	AssemblyWriter();

public:
	void write(std::ostream& stream, const ir::Module& m);

};

}

}

