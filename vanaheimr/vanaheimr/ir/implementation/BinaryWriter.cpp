/*! \file   BinaryWriter.cpp
	\date   Saturday February 25, 2012
	\author Sudnya Diamos <mailsudnya@gmail.com>
	\brief  The implementation file for the helper class that traslates compiler IR to a binary.
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/BinaryWriter.h>
// Forward Declarations

/*! \brief The wrapper namespace for Vanaheimr */
namespace vanaheimr
{
/*! \brief A namespace for the internal representation */
namespace ir
{
    BinaryWriter(const Module& inputModule) : m_Module(inputModule)
    {
    }

    void writeBinary(std::ostream& binary)
    {
        populateHeader();
        populateSymbolTable();
        populateInstructions();
        populateData();
        populateStringTable();
        linkSymbols();

        binary.write(&m_header, sizeof(Header));
        binary.write(m_symbolTable.data(), getSymbolTableSize());
        binary.write(m_instructions.data(), getInstructionStreamSize());
        binary.write(m_data.data(), getDataSize());
        binary.write(m_stringTable.data(), getStringTableSize());

    }

}//namespace ir

}//namespace vanaheimr
