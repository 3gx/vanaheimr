/*! \file   Binary.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Friday September 9, 2011
	\brief  The source file the IR Binary class.
*/

// Archaeopteryx Includes
#include <archaeopteryx/ir/interface/Binary.h>

#include <archaeopteryx/util/interface/File.h>

#include <archaeopteryx/util/interface/debug.h>
#include <archaeopteryx/util/interface/cstring.h>

// Vanaheimr Includes
#include <vanaheimr/asm/interface/Instruction.h>


#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace archaeopteryx
{

namespace ir
{

__device__ Binary::Binary(const char* filename)
: _file(0), _ownedFile(0)
{
	_ownedFile = new util::File(filename, "r");
	
	_file = _ownedFile;

	_loadHeader();
}

__device__ Binary::Binary(File* file)
: _file(file), _ownedFile(0)
{
	_loadHeader();
}

__device__ Binary::~Binary()
{
	for(unsigned int c = 0; c != _header.codePages; ++c)
	{
		delete[] _codeSection[c];
	}
	
	for(unsigned int d = 0; d != _header.dataPages; ++d)
	{
		delete[] _dataSection[d];
	}
	
	for(unsigned int s = 0; s != _header.stringPages; ++s)
	{
		delete[] _stringSection[s];
	}
	
	delete[] _symbolTable;
	delete[] _codeSection;
	delete[] _dataSection;
	delete[] _stringSection;
	
	delete _ownedFile;
}

__device__ Binary::SymbolTableEntry* Binary::findSymbol(const char* name)
{
	_loadSymbolTable();
	
	for(unsigned int i = 0; i < _header.symbols; ++i)
	{
		SymbolTableEntry* symbol = _symbolTable + i;
			
		if(_strcmp(symbol->stringOffset, name) != 0)
		{
			return symbol;
		}
	}
	
	return 0;
}

__device__ void Binary::copyCode(InstructionContainer* code, PC pc,
	unsigned int instructions)
{
	const size_t instructionsPerPage = sizeof(PageDataType) /
		sizeof(InstructionContainer);
	
	size_t page       = pc / instructionsPerPage;
	size_t pageOffset = pc % instructionsPerPage;
	
	device_report("Copying %d instructions at PC %d\n", instructions, pc);

	while(instructions > 0)
	{
		size_t instructionsInThisPage =
			util::min((size_t)(instructionsPerPage - pageOffset),
				(size_t)instructions);
	
		device_report(" copying %d instructions from page %d\n", 
			(int)instructionsInThisPage, (int)page);
		PageDataType* pageData = getCodePage(code_begin() + page);
		device_assert(pageData != 0);

		InstructionContainer* container =
			reinterpret_cast<InstructionContainer*>(pageData);
	
		util::memcpy(code, container + pageOffset,
			sizeof(InstructionContainer) * instructionsInThisPage);
	
		instructions -= instructionsInThisPage;
		pageOffset    = 0;
		page         += 1;

		device_report("  %d instructions are remaining\n", instructions);
	}
}

__device__ bool Binary::containsFunction(const char* name)
{
	SymbolTableEntry* symbol = findSymbol(name);
	
	if(symbol == 0) return false;
	
	return symbol->type == SymbolTableEntry::FunctionType;
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
	
	device_assert(symbol->type == SymbolTableEntry::FunctionType);
	
	page   = code_begin() + _getCodePageId(symbol->offset);
	offset = _getCodePageOffset(symbol->offset);
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
	
	device_assert(symbol->type == SymbolTableEntry::VariableType);
	
	page   = data_begin() + _getDataPageId(symbol->offset);
	offset = _getDataPageOffset(symbol->offset);
}

__device__ util::string Binary::getSymbolDataAsString(const char* symbolName)
{
	device_report("   getting data for symbol '%s'\n", symbolName);
	
	SymbolTableEntry* symbol = findSymbol(symbolName);
	
	device_assert(symbol != 0);

	device_assert(symbol->type == SymbolTableEntry::VariableType);
	
	util::string result(symbol->size, '\0');
	
	_strcpy((char*)result.data(), symbol->offset);
	
	return result;
}

__device__ Binary::PC Binary::findFunctionsPC(const char* name)
{
	page_iterator page  = 0;
	unsigned int offset = 0;

	findFunction(page, offset, name);
	
	const size_t instructionsPerPage = sizeof(PageDataType) /
		sizeof(InstructionContainer);
	
	return instructionsPerPage * (page - code_begin()) + offset;
}

__device__ Binary::page_iterator Binary::code_begin()
{
	return _codeSection;
}

__device__ Binary::page_iterator Binary::code_end()
{
	return _codeSection + _header.codePages;
}

__device__ Binary::page_iterator Binary::data_begin()
{
	return _dataSection;
}

__device__ Binary::page_iterator Binary::data_end()
{
	return _dataSection + _header.dataPages;
}

__device__ Binary::page_iterator Binary::string_begin()
{
	return _stringSection;
}

__device__ Binary::page_iterator Binary::string_end()
{
	return _stringSection + _header.stringPages;
}

__device__ Binary::PageDataType* Binary::getCodePage(page_iterator page)
{
	if(*page == 0)
	{
		// TODO lock the page
		
		size_t offset = _getCodePageOffset(page);

		device_report("Loading code page (%p) at offset (%p) now...\n",
			page, offset);

		_file->seekg(offset);
		*page = (PageDataType*)new PageDataType;
		_file->read(*page, sizeof(PageDataType));
	}
	
	return *page;
}

__device__ Binary::PageDataType* Binary::getDataPage(page_iterator page)
{
	if(*page == 0)
	{
		// TODO lock the page
	
		size_t offset = _getDataPageOffset(page);

		device_report("Loading data page (%p) at offset (%p) now...\n",
			page, offset);

		_file->seekg(offset);
		*page = (PageDataType*)new PageDataType;
		_file->read(*page, sizeof(PageDataType));
	}
	
	return *page;
}

__device__ Binary::PageDataType* Binary::getStringPage(page_iterator page)
{
	device_assert(page < string_end());

	if(*page == 0)
	{
		// TODO lock the page
	
		size_t offset = _getStringPageOffset(page);

		device_report("Loading string page (%p) at offset (%p) now...\n",
			page, offset);

		_file->seekg(offset);
		*page = (PageDataType*)new PageDataType;
		_file->read(*page, sizeof(PageDataType));
	}
	
	return *page;
}


__device__ void Binary::_loadHeader()
{
	_file->read(&_header, sizeof(Header));
	
	device_report("Loading header (%p magic)\n", _header.magic);
	
	device_assert(_header.magic == Header::MagicNumber);
	
	_dataSection   = new PagePointer[_header.dataPages];
	_codeSection   = new PagePointer[_header.codePages];
	_stringSection = new PagePointer[_header.stringPages];

	_symbolTable = 0;

	util::memset(_dataSection,   0, _header.dataPages   * sizeof(PagePointer));
	util::memset(_codeSection,   0, _header.codePages   * sizeof(PagePointer));
	util::memset(_stringSection, 0, _header.stringPages * sizeof(PagePointer));
	
	device_report("Loaded binary (%d data pages, %d code pages, "
		"%d symbols, %d string pages)\n", _header.dataPages, _header.codePages,
		_header.symbols, _header.stringPages);
}

__device__ void Binary::_loadSymbolTable()
{
	if(_header.symbols == 0) return;
	if(_symbolTable != 0)    return;

	device_report(" Loading symbol table now.\n");

	_symbolTable = new SymbolTableEntry[_header.symbols];
	
	device_report("  symbol table offset %d.\n", (int)_header.symbolOffset);
	device_assert(_file != 0);

	_file->seekg(_header.symbolOffset);

	device_report("  loading symbol table now.\n");

	_file->read(_symbolTable, _header.symbols * sizeof(SymbolTableEntry));

	device_report("   loaded %d symbols...\n", _header.symbols);
}

__device__ size_t Binary::_getCodePageOffset(page_iterator page)
{
	return _header.codeOffset +	(page - code_begin()) * sizeof(PageDataType);
}

__device__ size_t Binary::_getDataPageOffset(page_iterator page)
{
	return _header.dataOffset + (page - data_begin()) * sizeof(PageDataType);
}

__device__ size_t Binary::_getStringPageOffset(page_iterator page)
{
	return _header.stringsOffset +
		(page - string_begin()) * sizeof(PageDataType);
}

__device__ int Binary::_strcmp(unsigned int stringTableOffset,
	const char* string)
{
	page_iterator page  = string_begin() + _getStringPageId(stringTableOffset);
	unsigned int offset = _getStringPageOffset(stringTableOffset);

	device_report("comparing string at offset %d against '%s'\n",
		stringTableOffset, string);
	
	for(; page != string_end(); ++page, offset = 0)
	{
		const char* data = (const char*)*getStringPage(page);
		
		for(; offset != sizeof(PageDataType); ++offset, ++string)
		{
			if(data[offset] != *string)
			{
				return -1;
			}
			
			if(data[offset] == '\0')
			{
				if(*string == '\0')
				{
					return 0;
				}
				
				return -1;
			}
			else if(*string == '\n')
			{
				return -1;
			}
		}
	}
	
	return 0;
}

__device__ void Binary::_strcpy(char* string, unsigned int stringTableOffset)
{
	page_iterator page  = string_begin() + _getStringPageId(stringTableOffset);
	unsigned int offset = _getStringPageOffset(stringTableOffset);

	for(; page != string_end(); ++page, offset = 0)
	{
		const char* data = (const char*)*getStringPage(page);
		
		for(; offset != sizeof(PageDataType); ++offset, ++string)
		{
			if(data[offset] == '\0')
			{
				return;
			}
			
			*string = data[offset];
		}
	}
}

__device__ unsigned int Binary::_getCodePageId(size_t offset)
{
	size_t codeOffset = offset - _header.codeOffset;
	
	return codeOffset / sizeof(PageDataType);
}

__device__ unsigned int Binary::_getCodePageOffset(size_t offset)
{
	size_t codeOffset = offset - _header.codeOffset;
	
	return codeOffset % sizeof(PageDataType);
}

__device__ unsigned int Binary::_getDataPageId(size_t offset)
{
	size_t dataOffset = offset - _header.dataOffset;
	
	return dataOffset / sizeof(PageDataType);
}

__device__ unsigned int Binary::_getDataPageOffset(size_t offset)
{
	size_t dataOffset = offset - _header.dataOffset;
	
	return dataOffset % sizeof(PageDataType);
}

__device__ unsigned int Binary::_getStringPageId(size_t offset)
{
	device_assert(offset >= _header.stringsOffset);

	size_t stringsOffset = offset - _header.stringsOffset;
	
	return stringsOffset / sizeof(PageDataType);
}

__device__ unsigned int Binary::_getStringPageOffset(size_t offset)
{
	size_t stringsOffset = offset - _header.stringsOffset;
	
	return stringsOffset % sizeof(PageDataType);
}

}

}

