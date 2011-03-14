/*! \file   Binary.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Sunday February 27, 2011
	\brief  The header file the IR Binary class.
*/

#pragma once

namespace ir
{

/*! \brief A class representing a VIR binary */
class Binary
{
public:
	/*! \brief 32-bit unsigned int */
	typedef unsigned int uint32;
	/*! \brief A 32-KB page */
	typedef uint32[1 << 13] PageDataType;
	
	/*! \brief A symbol type */
	enum SymbolType
	{
		VariableSymbolType = 0x1,
		FunctionSymbolType = 0x2,
		InvalidSymbolType  = 0x0
	};

	/*! \brief A table mapping symbols to pages and offsets */
	class SymbolTableEntry
	{
	public:
		/*! \brief The type of symbol */
		uint32 type;
		/*! \brief The offset in the string table of the name */
		uint32 stringTableOffset;
		/*! \brief The page id it is stored in */
		uint32 pageId;
		/*! \brief The offset within the page */
		uint32 pageOffset;
	};

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

};

}

