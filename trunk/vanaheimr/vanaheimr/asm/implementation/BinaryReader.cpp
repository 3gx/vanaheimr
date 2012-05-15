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

ir::Module* BinaryReader::read(std::istream& stream, const std::string& name)
{
	_readHeader(stream);
	_readDataSection(stream);
	_readStringTable(stream);
	_readSymbolTable(stream);
	_readInstructions(stream);

	ir::Module* module = new ir::Module(name,
		compiler::Compiler::getSingleton());

	_initializeModule(*module);
	
	return module;
}

void BinaryReader::_readHeader(std::istream& stream)
{
	stream.read((char*)&_header, sizeof(BinaryHeader));

	if(stream.gcount() != sizeof(BinaryHeader))
	{
		throw std::runtime_error("Failed to read binary "
			"header, hit EOF.");
	}
}

void BinaryReader::_readDataSection(std::istream& stream)
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

void BinaryReader::_readStringTable(std::istream& stream)
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

void BinaryReader::_readSymbolTable(std::istream& stream)
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

void BinaryReader::_readInstructions(std::istream& stream)
{
	size_t dataSize = BinaryHeader::PageSize * _header.codePages;
	size_t sizeInInstructions = (dataSize + sizeof(InstructionContainer) - 1) /
		sizeof(InstructionContainer);

	_instructions.resize(sizeInInstructions);

	// TODO obey page alignment
	stream.read((char*) _instructions.data(), dataSize);

	if(stream.gcount() != dataSize)
	{
		throw std::runtime_error("Failed to read code section, hit EOF.");
	}
}

void BinaryReader::_initializeModule(ir::Module& m) const
{
	_loadGlobals(m);
	_loadFunctions(m);
}

void BinaryReader::_loadGlobals(ir::Module& m) const
{
	for(auto symbol : _symbolTable)
	{
		if(symbol.type != archaeopteryx::ir::Binary::VariableSymbolType)
		{
			continue;
		}

		auto global = m.newGlobal(_getSymbolName(symbol), _getSymbolType(symbol),
			_getSymbolLinkage(symbol));

		if(_hasInitializer(symbol))
		{
			global->setInitializer(_createConstant(symbol));
		}
	}
}

void BinaryReader::_loadFunctions(ir::Module& m)
{
	_findBasicBlocks(m);
	
	for(auto symbol : _symbolTable)
	{
		if(symbol.type != archaeopteryx::ir::Binary::FunctionSymbolType)
		{
			continue;
		}

		ir::Module::iterator function = m.newFunction(_getSymbolName(symbol),
			_getSymbolLinkage(symbol));
		
		BasicBlockOffsetVector blocks =
			_getBasicBlocksInFunction(function->name());
		
		for(auto blockOffset : blocks)
		{
			ir::Function::iterator block = function->newBlock(
				function->end(), blockOffset.name);
			
			for(unsigned int i = blockOffset.begin; i != blockOffset.end; ++i)
			{
				_addInstruction(block, _instructions[i]);
			}
		}
	}
}

std::string BinaryReader::_getSymbolName(symbol_iterator symbol) const
{
	return std::string((char*)_stringTable.data() + symbol->stringOffset);
}

ir::Type* BinaryReader::_getSymbolType(symbol_iterator symbol) const
{

}

ir::Variable::Linkage BinaryReader::_getSymbolLinkage(symbol_iterator symbol) const
{
	
}

bool BinaryReader::_hasInitializer(symbol_iterator symbol) const;
ir::Constant* BinaryReader::_getInitializer(symbol_iterator symbol) const;

void BinaryReader::_findBasicBlocks(ir::Module& m);
BinaryReader::BasicBlockDescriptorVector
	BinaryReader::_getBasicBlocksInFunction(const std::string& name);

void BinaryReader::_addInstruction(ir::Function::iterator block,
	const InstructionContainer& container);


}

}

