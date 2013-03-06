/*! \file   TranslationTable.cpp
	\date   Wednesday February 1, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The source file for the TranslationTable class.
*/

// Vanaheimr Includes
#include <vanaheimr/machine/interface/TranslationTable.h>
#include <vanaheimr/machine/interface/TranslationTableEntry.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

// Standard Library Includes
#include <map>

namespace vanaheimr
{

namespace machine
{

class TranslationTableMap
{
public:
	typedef std::map<std::string, TranslationTableEntry> Map;

public:
	Map opcodeToTranslation;

};

TranslationTable::TranslationTable()
: _translations(new TranslationTableMap)
{

}

TranslationTable::~TranslationTable()
{
	delete _translations;
}

TranslationTable::MachineInstructionVector
	TranslationTable::translateInstruction(const ir::Instruction*) const
{
	assertM(false, "Not implemented");
	
	MachineInstructionVector translatedInstructions;

	return translatedInstructions;
}

const TranslationTableEntry* TranslationTable::getTranslation(
	const std::string& name) const
{
	auto translation = _translations->opcodeToTranslation.find(name);
	
	if(translation == _translations->opcodeToTranslation.end())
	{
		return nullptr;
	}
	
	return &translation->second;
}

void TranslationTable::addTranslation(const TranslationTableEntry& entry)
{
	assert(_translations->opcodeToTranslation.count(entry.name) == 0);

	_translations->opcodeToTranslation.insert(
		std::make_pair(entry.name, entry));
}

}

}


