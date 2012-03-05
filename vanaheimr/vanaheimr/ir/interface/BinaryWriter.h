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
    typedef std::vector<SymbolTableEntry> SymbolTableEntryVector;
    typedef SymbolTableEntryVector::iterator symbol_iterator;

public:
    static const unsigned int PageSize = (1 << 15); // 32 KB

public:
    BinaryWriter(const Module& inputModule);
    void writeBinary(std::ostream& binary);

private:
    Header m_header;
    std::vector<InstructionContainer> m_instructions;
    std::vector<char> m_data;
    std::vector<SymbolTableEntry> m_symbolTable;
    std::vector<char> m_strings;

private:
    void populateHeader();
    void populateInstructions();
    void populateData();
    void linkSymbols();

private:
    InstructionContainer convertToContainer(Instruction); 

    size_t getHeaderOffset() const;
    size_t getInstructionOffset() const;
    size_t getDataOffset() const;
    size_t getSymbolTableOffset() const;
    size_t getStringOffset() const;

}//class BinaryWriter ends

}//namespace ir

}//namespace vanaheimr
