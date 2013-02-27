/*! \file   TranslationTable.h
	\date   Wednesday February 1, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the TranslationTable class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

// Forward Declarations
namespace vanaheimr { namespace machine { class TranslationTableEntry; } }
namespace vanaheimr { namespace machine { class TranslationTableMap;   } }
namespace vanaheimr { namespace machine { class Instruction;           } }
namespace vanaheimr { namespace ir      { class Instruction;           } }

namespace vanaheimr
{

namespace machine
{

/*! \brief A collection of rules for performing instruction selection */
class TranslationTable
{
public:
	TranslationTable();
	~TranslationTable();

public:
	TranslationTable(const TranslationTable&) = delete;
	TranslationTable& operator=(const TranslationTable&) = delete;

public:
	machine::Instruction* translateInstruction(const ir::Instruction*);

public:
	const TranslationTableEntry* getTranslation(const std::string& name) const;

public:
	void addTranslation(const TranslationTableEntry& entry);

private:
	TranslationTableMap* _translations;

};

}

}


