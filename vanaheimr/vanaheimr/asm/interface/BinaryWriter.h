/*! 	\file   BinaryWriter.h
	\date   Saturday February 25, 2012
	\author Sudnya Diamos <mailsudnya@gmail.com>
	\brief  The header file for the helper class that traslates compiler IR to a binary.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/asm/interface/BinaryHeader.h>

#include <vanaheimr/asm/interface/SymbolTableEntry.h>

// Archaeopteryx Includes
#include <archaeopteryx/ir/interface/Instruction.h>

// Standard Library Includes
#include <vector>
#include <ostream>

// Forward Declarations
namespace vanaheimr { namespace ir { class Module;      } }
namespace vanaheimr { namespace ir { class Instruction; } }

/*! \brief The wrapper namespace for Vanaheimr */
namespace vanaheimr
{
/*! \brief A namespace for the internal representation */
namespace as
{

/*! \brief Represents a single compilation unit. */
class BinaryWriter
{
public:
	typedef vanaheimr::ir::Instruction Instruction;
	typedef archaeopteryx::ir::InstructionContainer InstructionContainer;
	typedef std::vector<SymbolTableEntry> SymbolTableEntryVector;
	typedef SymbolTableEntryVector::iterator symbol_iterator;

public:
	static const unsigned int PageSize = (1 << 15); // 32 KB

public:
	BinaryWriter();
	void write(std::ostream& binary, const ir::Module& inputModule);

private:
	void populateHeader();
	void populateInstructions();
	void populateData();
	void linkSymbols();

private:
	InstructionContainer convertToContainer(const Instruction&); 

	size_t getHeaderOffset() const;
	size_t getInstructionOffset() const;
	size_t getDataOffset() const;
	size_t getSymbolTableOffset() const;
	size_t getStringTableOffset() const;

	size_t getSymbolTableSize() const;
	size_t getInstructionStreamSize() const;
	size_t getDataSize() const;
	size_t getStringTableSize() const;

private:
	typedef std::vector<InstructionContainer> InstructionVector;
	typedef std::vector<char>                 DataVector;
	typedef std::vector<SymbolTableEntry>     SymbolVector;

private:
	const ir::Module*  m_module;
	
	BinaryHeader      m_header;
	InstructionVector m_instructions;
	DataVector        m_data;
	SymbolVector      m_symbolTable;
	DataVector        m_stringTable;
};

}

}

