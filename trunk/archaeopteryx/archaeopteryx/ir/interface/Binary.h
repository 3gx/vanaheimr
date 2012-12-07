/*! \file   Binary.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Sunday February 27, 2011
	\brief  The header file the IR Binary class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/util/interface/IntTypes.h>

// Forward Declarations
namespace archaeopteryx { namespace util { class File;                 } }
namespace vanaheimr     { namespace as   { class InstructionContainer; } }

namespace archaeopteryx
{

namespace ir
{

/*! \brief A class representing a VIR binary, lazy loading is handled here */
class Binary
{
public:
	/*! \brief An instruction object stored in the binary */
	typedef vanaheimr::as::InstructionContainer InstructionContainer;

	/*! \brief a 64-bit program counter */
	typedef uint64_t PC;
	/*! \brief a file handle */
	typedef util::File File;

	/*! \brief A binary header */
	class Header
	{
	public:
		uint32_t dataPages;
		uint32_t codePages;
		uint32_t symbols;
		uint32_t strings;
	};

	/*! \brief A 32-KB page */
	typedef uint32_t PageDataType[1 << 13];
	
	/*! \brief A symbol type */
	enum SymbolType
	{
		VariableSymbolType   = 0x1,
		FunctionSymbolType   = 0x2,
		ArgumentSymbolType   = 0x3,
		BasicBlockSymbolType = 0x4,
		InvalidSymbolType    = 0x0
	};

	/*! \brief A symbol attribute */
	enum SymbolAttribute
	{
		InvalidAttribute = 0x0,
	};

	/*! \brief A table mapping symbols to pages and offsets */
	class SymbolTableEntry
	{
	public:
		/*! \brief The type of symbol */
		uint32_t type;
		/*! \brief The offset in the string table of the name */
		uint32_t stringTableOffset;
		/*! \brief The page id it is stored in */
		uint32_t pageId;
		/*! \brief The offset within the page */
		uint32_t pageOffset;
		/*! \brief The set of attributes */
		uint64_t attributes;
	};

	/*! \brief A symbol table iterator */
	typedef SymbolTableEntry* symbol_table_iterator;

	/*! \brief A page iterator */
	typedef PageDataType** page_iterator;

public:
	/*! \brief Construct a binary from a file name */
	__device__ Binary(const char* filename);
	/*! \brief Construct a binary from an open file */
	__device__ Binary(File* file);
	/*! \brief Destroy the binary, free all memory */
	__device__ ~Binary();

public:
	/*! \brief Get a particular code page */
	__device__ PageDataType* getCodePage(page_iterator page);
	/*! \brief Get a pointer to a particular data page */
	__device__ PageDataType* getDataPage(page_iterator page);

	
	/*! \brief Does a named funtion exist? */
	__device__ bool containsFunction(const char* name);
	
	/*! \brief Find a symbol by name */
	__device__ SymbolTableEntry* findSymbol(const char* name);
	/*! \brief Find a function by name */
	__device__ void findFunction(page_iterator& page, unsigned int& offset,
		const char* name);
	/*! \brief Find a variable by name */
	__device__ void findVariable(page_iterator& page, unsigned int& offset,
		const char* name);

public:
	/*! \brief Get PC */
	__device__ PC findFunctionsPC(const char* name);

public:
	/*! \brief Get an iterator to the first code page */
	__device__ page_iterator code_begin();
	/*! \brief Get an iterator to one past the last code page */
	__device__ page_iterator code_end();

	/*! \brief Get an iterator to the first data page */
	__device__ page_iterator data_begin();
	/*! \brief Get an iterator to one past the last data page */
	__device__ page_iterator data_end();

public:
	/*! \brief Copy code from a PC */
	__device__ void copyCode(InstructionContainer* code, PC pc,
		unsigned int instructions);

public:
	/*! \brief The number of pages in the data section */
	unsigned int dataPages;
	/*! \brief The list of data pages, lazily allocated */
	PageDataType** dataSection;
	/*! \brief The number of pages in the code section */
	unsigned int codePages;
	/*! \brief The list of instruction pages, lazily allocated */
	PageDataType** codeSection;
	/*! \brief The number of symbol table entries */
	unsigned int symbolTableEntries;
	/*! \brief The actual symbol table */
	SymbolTableEntry* symbolTable;
	/*! \brief The string table */
	char* stringTable;
	/*! \brief The number of string table entries */
	unsigned int stringTableEntries;

private:
	/*! \brief Get an offset in the file for a specific code page */
	__device__ size_t _getCodePageOffset(page_iterator page);
	/*! \brief Get an offset in the file for a specific data page */
	__device__ size_t _getDataPageOffset(page_iterator page);
	/*! \brief Load the symbol and string tables */
	__device__ void _loadSymbolTable();

private:
	/*! \brief A handle to the file */
	File* _file;
	/*! \brief A handle to a file owned by this binary */
	File* _ownedFile;

};

}

}

