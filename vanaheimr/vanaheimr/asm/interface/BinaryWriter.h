/*! \file   BinaryWriter.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Sunday February 12, 2012
	\brief  The header file for the BinaryWriter class.
*/

#pragma once

// Forward Declarations
namespace vanaheimr{ namespace ir { class Module; } }

namespace vanaheimr
{

namespace asm
{

/*! \brief Used to write a module to an object file */
class BinaryWriter
{
public:
	BinaryWriter();

public:
	void write(std::ostream& stream, const ir::Module& m);

};

}

}

