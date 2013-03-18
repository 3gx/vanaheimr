/*! \file   TranslationTable.cpp
	\date   Wednesday February 1, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The source file for the TranslationTable class.
*/

// Vanaheimr Includes
#include <vanaheimr/machine/interface/TranslationTable.h>
#include <vanaheimr/machine/interface/TranslationTableEntry.h>
#include <vanaheimr/machine/interface/Instruction.h>

#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/BasicBlock.h>
#include <vanaheimr/ir/interface/Instruction.h>
#include <vanaheimr/ir/interface/Constant.h>
#include <vanaheimr/ir/interface/Type.h>

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

typedef std::vector<ir::VirtualRegister*> RegisterVector;

static void mapOperands(ir::Instruction* newInstruction,
	const ir::Instruction* instruction,
	const TranslationTableEntry::Translation& translation,
	RegisterVector& temporaries)
{
	auto operands = instruction->writes;
	operands.insert(operands.end(), instruction->reads.begin(),
		instruction->reads.end());
	
	for(auto argument : translation.arguments)
	{
		if(argument.isImmediate())
		{
			if(argument.immediate->type()->isFloatingPoint())
			{
				auto floatConstant =
					static_cast<const ir::FloatingPointConstant*>(
					argument.immediate);
				
				if(argument.immediate->type()->isSinglePrecisionFloat())
				{
					newInstruction->reads.push_back(new ir::ImmediateOperand(
						floatConstant->asFloat(), newInstruction,
						argument.immediate->type()));
				}
				else
				{
					newInstruction->reads.push_back(new ir::ImmediateOperand(
						floatConstant->asDouble(), newInstruction,
						argument.immediate->type()));
				}
			}
			else
			{
				auto integerConstant = static_cast<const ir::IntegerConstant*>(
					argument.immediate);
				
				newInstruction->reads.push_back(new ir::ImmediateOperand(
					(uint64_t)(*integerConstant), newInstruction,
					argument.immediate->type()));
			}
		}
		else if(argument.isRegister())
		{
			auto operand = operands[argument.index]->clone();
			
			if(argument.isSource)
			{		
				newInstruction->reads.push_back(operand);
			}
			else
			{
				newInstruction->writes.push_back(operand);
			}
		}
		else
		{
			assert(argument.isTemporary());
			
			newInstruction->reads.push_back(
				new ir::RegisterOperand(temporaries[argument.index],
				newInstruction));
		}
	}
}

TranslationTable::MachineInstructionVector
	TranslationTable::translateInstruction(
	const ir::Instruction* instruction) const
{
	MachineInstructionVector translatedInstructions;

	auto translation = getTranslation(instruction->opcodeString());
	
	assert(translation != nullptr);

	// Create temporary registers
	RegisterVector temporaries;

	auto temps = translation->getTemporaries();
	
	auto function = instruction->block->function();
	
	for(auto temp : temps)
	{
		assert(temp.index == temporaries.size());
	
		temporaries.push_back(&*function->newVirtualRegister(temp.type));
	}
	
	// Translate instructions
	for(auto entry : translation->translations)
	{
		auto newInstruction = new Instruction(entry.operation);
		
		mapOperands(newInstruction, instruction, entry, temporaries);
	
		translatedInstructions.push_back(newInstruction);
	}

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


