/*! \file   Binary.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday September 9, 2011
	\brief  The source file the IR Binary class.
*/

// Archaeopteryx Includes
#include <archaeopteryx/ir/interface/Binary.h>
#include <archaeopteryx/ir/interface/Instruction.h>

#include <archaeopteryx/util/interface/File.h>
#include <archaeopteryx/util/interface/StlFunctions.h>

namespace ir
{

__device__ Binary::Binary(File* file)
{
	Header header;

	file->read(&header, sizeof(Header));

	dataPages          = header.dataPages;
	codePages          = header.codePages;
	symbolTableEntries = header.symbols;
	stringTableEntries = header.strings;
	
	dataSection = new PageDataType*[dataPages];
	codeSection = new PageDataType*[codePages];
	symbolTable = new SymbolTableEntry[symbolTableEntries];
	stringTable = new char*[stringTableEntries];

	std::memset(dataSection, 0, dataPages * sizeof(PageDataType*));
	std::memset(codeSection, 0, codePages * sizeof(PageDataType*));
	std::memset(symbolTable, 0, symbolTableEntries * sizeof(SymbolTableEntry));
	std::memset(stringTable, 0, stringTableEntries * sizeof(char*));
}

__device__ Binary::~Binary()
{
	for(unsigned int s = 0; s != stringTableEntries; ++s)
	{
		delete stringTable[s];
	}
	
	for(unsigned int c = 0; c != codePages; ++c)
	{
		delete[] codeSection[c];
	}
	
	for(unsigned int d = 0; d != dataPages; ++d)
	{
		delete[] dataSection[d];
	}
	
	delete[] stringTable;
	delete[] symbolTable;
	delete[] codeSection;
	delete[] dataSection;
}

__device__ Binary::PageDataType* Binary::getCodePage(page_iterator page)
{
	if(*page == 0)
	{
		_file->seekg(_getCodePageOffset(page));
		*page = (PageDataType*)new PageDataType;
		_file->read(*page, sizeof(PageDataType));
	}
	
	return *page;
}

__device__ Binary::PageDataType* Binary::getDataPage(page_iterator page)
{
	if(*page == 0)
	{
		_file->seekg(_getDataPageOffset(page));
		*page = (PageDataType*)new PageDataType;
		_file->read(*page, sizeof(PageDataType));
	}
	
	return *page;
}

__device__ Binary::SymbolTableEntry* Binary::findSymbol(const char* name)
{
	for(unsigned int i = 0; i < symbolTableEntries; ++i)
	{
		SymbolTableEntry* symbol = symbolTable + i;
		const char* symbolName   = *(symbol->stringTableOffset + stringTable);
	
		if(util::strcmp(symbolName, name) != 0)
		{
			return symbol;
		}
	}
	
	return 0;
}

__device__ void Binary::findFunction(page_iterator& page, unsigned int& offset,
	const char* name)
{
	SymbolTableEntry* symbol = findSymbol(name);
	
	if(symbol == 0)
	{
		page   = 0;
		offset = 0;
		
		return;
	}
	
	util::device_assert(symbol->type == FunctionSymbolType);
	
	page   = codeSection + symbol->pageId;
	offset = symbol->pageOffset;
}

__device__ void Binary::findVariable(page_iterator& page, unsigned int& offset,
	const char* name)
{
	SymbolTableEntry* symbol = findSymbol(name);
	
	if(symbol == 0)
	{
		page   = 0;
		offset = 0;
		
		return;
	}
	
	util::device_assert(symbol->type == VariableSymbolType);
	
	page   = dataSection + symbol->pageId;
	offset = symbol->pageOffset;
}

__device__ Binary::page_iterator Binary::code_begin()
{
	return codeSection;
}

__device__ Binary::page_iterator Binary::code_end()
{
	return codeSection + codePages;
}

__device__ Binary::page_iterator Binary::data_begin()
{
	return dataSection;
}

__device__ Binary::page_iterator Binary::data_end()
{
	return dataSection + dataPages;
}

__device__ void Binary::copyCode(ir::InstructionContainer* code, PC pc,
	unsigned int instructions)
{
	const size_t instructionsPerPage = sizeof(PageDataType) /
		sizeof(ir::InstructionContainer);
	
	size_t page       = pc / instructionsPerPage;
	size_t pageOffset = pc % instructionsPerPage;
	
	while(instructions > 0)
	{
		size_t instructionsInThisPage =
			util::min(instructionsPerPage - pageOffset, (size_t)instructions);
	
		PageDataType* pageData = getCodePage(code_begin() + page);
		util::device_assert(pageData != 0);

		ir::InstructionContainer* instructions =
			reinterpret_cast<ir::InstructionContainer*>(pageData);
	
		std::memcpy(code, instructions + pageOffset,
			sizeof(ir::InstructionContainer) * instructionsInThisPage);
	
		instructions -= instructionsInThisPage;
		pageOffset    = 0;
		page         += 1;
	}
}

size_t Binary::_getCodePageOffset(page_iterator page)
{
	return _getDataPageOffset(data_begin() + dataPages) +
		(page - code_begin()) * sizeof(PageDataType);
}

size_t Binary::_getDataPageOffset(page_iterator page)
{
	return sizeof(Header) + (page - data_begin()) * sizeof(PageDataType);
}


}

