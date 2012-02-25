/*! \file   BinaryWriter.h
	\date   Saturday February 25, 2012
	\author Sudnya Diamos <mailsudnya@gmail.com>
	\brief  The header file for the helper class that traslates compiler IR to a binary.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Module.h>
// Forward Declarations

/*! \brief The wrapper namespace for Vanaheimr */
namespace vanaheimr
{
/*! \brief A namespace for the internal representation */
namespace ir
{

/*! \brief Represents a single compilation unit. */
class BinaryWriter
{
public:
    BinaryWriter(const Module& inputModule);
    void writeBinary(std::ostream& binary);

private:
    void writeHeader();
}//class BinaryWriter ends

}//namespace ir

}//namespace vanaheimr
