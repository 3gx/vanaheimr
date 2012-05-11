/*! \file   BinaryReader.cpp
	\date   Monday May 7, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the BinaryReader class.
*/

// Vanaheimr Includes
#include <vanaheimr/asm/interface/BinaryReader.h>

#include <vanaheimr/compiler/interface/Compiler.h>

#include <vanaheimr/ir/interface/Module.h>

namespace vanaheimr
{

namespace as
{

ir::Module* BinaryReader::read(std::istream& stream)
{
	_readHeader(stream);
	_readDataSection(stream);
	_readStringTable(stream);
	_readSymbolTable(stream);
	_readInstructions(stream);

	ir::Module* module = new ir::Module(_getName(),
		compiler::Compiler::getSingleton());

	_initializeModule(*module);
	
	return module;
}

void ir::BinaryReader::_readHeader(std::istream& stream)
{
	stream.read((char*)&_header, sizeof(BinaryHeader));

	if(stream.gcount() != sizeof(BinaryHeader))
	{
		throw std::runtime_error("Failed to read binary "
			"header, hit EOF.");
	}
}

void ir::BinaryReader::_readDataSection(std::istream& stream)
{
	size_t dataSize = BinaryHeader::PageSize * _header.dataPages;

	stream.seekg(_header.dataOffset, std::ios::beg);

	_dataSection.resize(dataSize);

	stream.read((char*) _dataSection.data(), dataSize);

	if(stream.gcount() != dataSize)
	{
		throw std::runtime_error("Failed to read binary data section, hit
			EOF."); 
	}
}

void ir::BinaryReader::_readStringTable(std::istream& stream)
{
	size_t stringTableSize = BinaryHeader::PageSize * _header.stringPages;

	stream.seeekg(_header.stringOffset, std::ios::beg);

	_stringTable.resize(stringTableSize);

	stream.read((char*) _stringTable.data(), stringTableSize);

	if(stream.gcount() != stringTableSize)
	{
		throw std::runtime_error("Failed to read string table, hit EOF");
	}
}

void ir::BinaryReader::_readSymbolTable(std::istream& stream)
{
	size_t symbolTableSize = sizeof(SymbolTableEntry) * _header.symbols;

	stream.seekg(_header.symbolOffset, std::ios::beg);

	_symbolTable.resize(_header.symbols);

	stream.read((char*) _symbolTable.data(), symbolTableSize);

	if(stream.gcount() != symbolTableSize)
	{
		throw std::runtime_error("Failed to read symbol table, hit EOF");
	}
}

void ir::BinaryReader::_readInstructions(std::istream& stream)
{
	size_t dataSize = BinaryHeader::PageSize * _header.codePages;

	// TODO
}

void ir::BinaryReader::_initializeModule(ir::Module& m) const;

void ir::BinaryReader::_loadGlobals(ir::Module& m)   const;
void ir::BinaryReader::_loadFunctions(ir::Module& m) const;

std::string ir::BinaryReader::_getName() const;

}

}


