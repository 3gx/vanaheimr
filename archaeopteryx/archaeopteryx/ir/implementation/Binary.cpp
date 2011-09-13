/*! \file   Binary.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday September 9, 2011
	\brief  The source file the IR Binary class.
*/

// Archaeopteryx Includes
#include <archaeopteryx/interface/Binary.h>

namespace ir
{

__device__ Binary::Binary(File* file)
{
	Header header;

	file->read(&header, sizeof(Header));

	dataPages          = header->dataPages;
	codePages          = header->codePages;
	symbolTableEntries = header->symbols;
	stringTableEntries = header->strings;
	
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
	
	for(unsigned int s = 0; s != symbolTableEntries; ++s)
	{
		delete symbolTable[s];
	}
	
	for(unsigned int c = 0; c != codePages; ++c)
	{
		delete codeSection[c];
	}
	
	for(unsigned int d = 0; d != dataPages; ++d)
	{
		delete dataSection[d];
	}
	
	delete[] stringTable;
	delete[] symbolTable;
	delete[] codeSection;
	delete[] dataSection;
}

__device__ PageDataType* Binary::getCodePage(page_iterator page)
{
	if(*page == 0)
	{
		file->setg(getCodePageOffset(page));
		*page = new PageDataType;
		file->read(*page, sizeof(PageDataType));
	}
	
	return *page;
}

__device__ PageDataType* Binary::getDataPage(page_iterator page)
{
	if(*page == 0)
	{
		file->setg(getDataPageOffset(page));
		*page = new PageDataType;
		file->read(*page, sizeof(PageDataType));
	}
	
	return *page;
}

__device__ Binary::SymbolTableEntry* Binary::findSymbol(const char* name)
{
	for(unsigned int i = 0; i < symbolTableEntries; ++i)
	{
		SymbolTableEntry* symbol = symbolTable + i;
		const char* symbolName   = symbol->stringTableOffset + stringTable;
	
		if(std::strcmp(symbolName, name) != 0)
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
	
	device_assert(symbol->type == FunctionSymbolType);
	
	page   = codePages + symbol->pageId;
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
	
	device_assert(symbol->type == VariableSymbolType);
	
	page   = dataPages + symbol->pageId;
	offset = symbol->pageOffset;
}

__device__ page_iterator Binary::code_begin()
{
	return codeSection;
}

__device__ page_iterator Binary::code_end()
{
	return codeSection + codePages;
}

__device__ page_iterator Binary::data_begin()
{
	return dataSection;
}

__device__ page_iterator Binary::data_end()
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
			std::min(instructionsPerPage - pageOffset, instructions);
	
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

}

#include <archaeopteryx/ir/implementation/Binary.cpp>

