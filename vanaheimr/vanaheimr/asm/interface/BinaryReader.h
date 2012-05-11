/*! \file   BinaryReader.h
	\date   Monday May 7, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the BinaryReader class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/asm/interface/BinaryHeader.h>
#include <vanaheimr/asm/interface/SymbolTableEntry.h>

// Archaeopteryx Includes
#include <archaeopteryx/ir/interface/Instruction.h>

// Standard Library Includes
#include <istream>
#include <vector>

namespace vanaheimr { namespace ir { class Module; } }

namespace vanaheimr
{

namespace as
{

/*! \brief Reads in a vanaheimr bytecode file yielding a module. */
class BinaryReader
{
public:
	/*! \brief Attempts to read from a binary stream, returns a module */
	ir::Module* read(std::istream& stream);

private:
	typedef archaeopteryx::ir::InstructionContainer InstructionContainer;
	typedef std::vector<InstructionContainer>       InstructionVector;
	typedef std::vector<char>                       DataVector;
	typedef std::vector<SymbolTableEntry>           SymbolVector;

private:
	void _readHeader(std::istream& stream);
	void _readDataSection(std::istream& stream);
	void _readStringTable(std::istream& stream);
	void _readSymbolTable(std::istream& stream);
	void _readInstructions(std::istream& stream);

private:
	void _initializeModule(ir::Module& m) const;

	void _loadGlobals(ir::Module& m)   const;
	void _loadFunctions(ir::Module& m) const;

private:
	std::string _getName() const;

private:
	BinaryHeader _header;

	InstructionVector _instructions;
	DataVector        _dataSection;
	DataVector        _stringTable;
	SymbolVector      _symbolTable;

};

}

}


