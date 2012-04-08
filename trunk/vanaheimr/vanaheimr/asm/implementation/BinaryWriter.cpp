/*!	\file   BinaryWriter.cpp
	\date   Saturday February 25, 2012
	\author Sudnya Diamos <mailsudnya@gmail.com>
	\brief  The implementation file for the helper class that traslates compiler IR to a binary.
*/

// Vanaheimr Includes
#include <vanaheimr/asm/interface/BinaryWriter.h>

#include <vanaheimr/ir/interface/Module.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

/*! \brief The wrapper namespace for Vanaheimr */
namespace vanaheimr
{

/*! \brief A namespace for the internal representation */
namespace as
{

BinaryWriter::BinaryWriter()
: m_module(0)
{

}

void BinaryWriter::write(std::ostream& binary, const ir::Module& m)
{
	m_module = &m;

	populateData();
	populateInstructions();
	linkSymbols();
	
	populateHeader();

	binary.write((const char*)&m_header, sizeof(BinaryHeader));
	binary.write((const char*)m_symbolTable.data(), getSymbolTableSize());
	binary.write((const char*)m_instructions.data(), getInstructionStreamSize());
	binary.write((const char*)m_data.data(), getDataSize());
	binary.write((const char*)m_stringTable.data(), getStringTableSize());
}

void BinaryWriter::populateData()
{
	for (ir::Module::const_global_iterator i = m_module->global_begin(); i != m_module->global_end(); ++i)
	{
		ir::Constant::DataVector blob;
		
		if (i->hasInitializer())
		{
			const ir::Constant* initializer = i->initializer();
			SymbolTableEntry temp;
			temp.type = 0x1;
			temp.attributes = 0x0;
			temp.stringOffset = m_stringTable.size();
			std::copy(i->name().begin(), i->name().end(), std::back_inserter(m_stringTable));
			m_stringTable.push_back('\0');
			temp.offset = m_data.size();//needs fix to offset within file instead of datasection
			m_symbolTable.push_back(temp);
			blob = initializer->data();
		}
		else
		{
			blob.resize(i->bytes());
		}

		std::copy(blob.begin(), blob.end(), std::back_inserter(m_data));
	}
}

void BinaryWriter::populateInstructions()
{
	for (ir::Module::const_iterator function = m_module->begin(); function != m_module->end(); ++function)
	{
		SymbolTableEntry temp;
		temp.type = 0x2;
		temp.attributes = 0x0;
		temp.stringOffset = m_stringTable.size();
		std::copy(function->name().begin(), function->name().end(), std::back_inserter(m_stringTable));
		m_stringTable.push_back('\0');
		temp.offset = m_instructions.size() * sizeof(InstructionContainer);
		m_symbolTable.push_back(temp);
		
		for (ir::Function::const_iterator bb = function->begin(); bb != function->end(); ++bb)
		{
			for (ir::BasicBlock::const_iterator inst = bb->begin(); inst != bb->end(); ++inst)
			{
				m_instructions.push_back(convertToContainer(**inst));
			}
		} 
	}
}

void BinaryWriter::linkSymbols()
{
	for (symbol_iterator symb = m_symbolTable.begin(); symb != m_symbolTable.end(); ++symb)
	{
		if (symb->type == 1)
		{
			symb->offset += getInstructionOffset();
		}
		else if (symb->type == 2)
		{
			symb->offset += getDataOffset();
		}
	}
}

void BinaryWriter::populateHeader()
{
	m_header.dataPages     = (m_data.size() + PageSize - 1) / PageSize; 
	m_header.codePages     = ((m_instructions.size()*sizeof(InstructionContainer)) + PageSize - 1) / PageSize;
	m_header.symbols       = m_symbolTable.size(); 
	m_header.stringPages   = (m_stringTable.size() + PageSize - 1) / PageSize;
	m_header.dataOffset    = getDataOffset();
	m_header.codeOffset    = getInstructionOffset();
	m_header.symbolOffset  = getSymbolTableOffset();
	m_header.stringsOffset = getStringTableOffset();
}

size_t BinaryWriter::getHeaderOffset() const
{
	return 0;
}

size_t BinaryWriter::getInstructionOffset() const
{
	return sizeof(m_header);
}

size_t BinaryWriter::getDataOffset() const
{
	return getInstructionStreamSize() + getInstructionOffset();
}

size_t BinaryWriter::getSymbolTableOffset() const
{
	return getDataSize() + getDataOffset();
}

size_t BinaryWriter::getStringTableOffset() const
{
	 return getSymbolTableSize() + getSymbolTableOffset();
}

size_t BinaryWriter::getSymbolTableSize() const
{
	return m_symbolTable.size() * sizeof(SymbolTableEntry);
}

size_t BinaryWriter::getInstructionStreamSize() const
{
	return m_instructions.size() * sizeof(InstructionContainer);
}

size_t BinaryWriter::getDataSize() const
{
	return m_data.size();
}

size_t BinaryWriter::getStringTableSize() const
{
	return m_stringTable.size();
}

BinaryWriter::InstructionContainer BinaryWriter::convertToContainer(const Instruction& instruction)
{
	InstructionContainer container;

	assertM(false, "Not implemented.");

	return container;
}

}

}

