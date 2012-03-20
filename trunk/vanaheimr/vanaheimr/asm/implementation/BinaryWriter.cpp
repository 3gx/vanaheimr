/*! \file   BinaryWriter.cpp
	\date   Saturday February 25, 2012
	\author Sudnya Diamos <mailsudnya@gmail.com>
	\brief  The implementation file for the helper class that traslates compiler IR to a binary.
*/

// Vanaheimr Includes
#include <vanaheimr/asm/interface/BinaryWriter.h>

// Forward Declarations

/*! \brief The wrapper namespace for Vanaheimr */
namespace vanaheimr
{
/*! \brief A namespace for the internal representation */
namespace as
{
BinaryWriter(const Module& inputModule) : m_module(inputModule)
{
}

void writeBinary(std::ostream& binary)
{
	populateData();
	populateInstructions();
	linkSymbols();
	
	populateHeader();

	binary.write(&m_header, sizeof(Header));
	binary.write(m_symbolTable.data(), getSymbolTableSize());
	binary.write(m_instructions.data(), getInstructionStreamSize());
	binary.write(m_data.data(), getDataSize());
	binary.write(m_stringTable.data(), getStringTableSize());

}

void populateData()
{
	for (Module::const_global_iterator i = m_module.global_begin(); i != m_module.global_end(); ++i)
	{
		Constant::DataVector blob;
		
		if (i->hasInitializer())
		{
			Constant* initializer = i->initializer();
			SymbolTableEntry temp;
			temp.type = 0x1;
			temp.attributes = 0x0;
			temp.stringOffset = m_strings.size();
			m_strings.push_back(i->name);
			temp.offset = m_data.size();//needs fix to offset within file instead of datasection
			m_symbolTableEntry.push_back(temp);
			blob = initializer->data();
		}
		else
		{
			blob.resize(i->bytes());
		}

		std::copy(blob.begin(), blob.end(), std::back_inserter(m_data));
	}
}

void populateInstructions()
{
	for (Module::iterator function = m_module.begin(); function != m_module.end(); ++function)
	{
	   SymbolTableEntry temp;
	   temp.type = 0x2;
	   temp.attributes = 0x0;
	   temp.stringOffset = m_strings.size();
	   m_strings.push_back(function->name());
	   temp.offset = m_instructions.size() * sizeof(InstructionContainer);
	   m_symbolTableEntry.push_back(temp);
	   for (Function::iterator bb = function->begin(); bb != function->end(); ++bb)
	   {
		  for (BasicBlock::iterator inst = bb->begin(); inst != bb->end(); ++inst)
		  {
			  m_instructions.push_back(convertToContainer(*inst));
		  }
	   } 
	}
}

void linkSymbols()
{
	for (symbol_iterator symb = m_symbolTable.begin(); symb != m_symbolTable.end(); ++symb)
	{
		if (symb->type == 1)
		{
			symb->offset += getInstructionOffset();
		} else if (symb->type == 2)
		{
			symb->offset += getDataOffset();
		}
	}
}

void populateHeader()
{
	m_header.dataPages    = (m_data.size() + PageSize - 1) / PageSize; 
	m_header.codePages    = ((m_instructions.size()*sizeof(InstructionContainer)) + PageSize - 1) / PageSize;
	m_header.symbols      = m_symbolTable.size(); 
	m_header.stringPages  = (m_strings.size() + PageSize - 1) / PageSize;
	m_header.dataOffset   = getDataOffset();
	m_header.codeOffset   = getCodeOffset();
	m_header.symbolOffset = getSymbolOffset();
	m_header.stringOffset = getStringOffset();
}

size_t getHeaderOffset() const
{
	return 0;
}

size_t getInstructionOffset() const
{
	return sizeof(m_header);
}

size_t getDataOffset() const
{
	return (m_instruction.size() * sizeof(InstructionContainer)) + getInstructionOffset();
}

size_t getSymbolTableOffset() const
{
	return (m_data.size() * sizeof(char)) + getDataOffset();
}

size_t getStringOffset() const
{
	 return (m_symbolTable.size() * sizeof(SymbolTableEntry)) + getSymbolTableOffset();
}

void convertToContainer(Instruction instruction)
{
}

}

}

