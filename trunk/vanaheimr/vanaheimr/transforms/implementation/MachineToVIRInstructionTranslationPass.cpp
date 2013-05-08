/*! \file   MachineToVIRInstructionTranslationPass.cpp
	\date   Tuesday May 6, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the MachineToVIRInstructionTranslationPass
		    class.
*/

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/MachineToVIRInstructionTranslationPass.h>
#include <vanaheimr/transforms/interface/MachineToVIRInstructionTranslationRule.h>

#include <vanaheimr/machine/interface/Instruction.h>

#include <vanaheimr/ir/interface/BasicBlock.h>
#include <vanaheimr/ir/interface/Instruction.h>

// Standard Library Includes
#include <cassert>

namespace vanaheimr
{

namespace transforms
{

MachineToVIRInstructionTranslationPass::MachineToVIRInstructionTranslationPass(
	const std::string& name)
: BasicBlockPass({}, name)
{

}

MachineToVIRInstructionTranslationPass::~MachineToVIRInstructionTranslationPass()
{
	clearTranslationRules();
}

MachineToVIRInstructionTranslationPass::MachineToVIRInstructionTranslationPass(
	const self& pass)
: BasicBlockPass({}, pass.name)
{
	for(auto rule : pass._translationRules)
	{
		addTranslationRule(rule.second);
	}
}

MachineToVIRInstructionTranslationPass&
	MachineToVIRInstructionTranslationPass::operator=(const self& pass)
{
	clearTranslationRules();

	for(auto rule : pass._translationRules)
	{
		addTranslationRule(rule.second);
	}
	
	return *this;
}

void MachineToVIRInstructionTranslationPass::addTranslationRule(
	const TranslationRule* rule)
{
	assert(_translationRules.count(rule->opcode) == 0);

	_translationRules.insert(std::make_pair(rule->opcode, rule->clone()));
}

void MachineToVIRInstructionTranslationPass::clearTranslationRules()
{
	for(auto rule : _translationRules)
	{
		delete rule.second;
	}
	
	_translationRules.clear();
}

void MachineToVIRInstructionTranslationPass::runOnBlock(BasicBlock& block)
{
	TranslationRule::InstructionVector newInstructions;

	// parallel-for-all
	for(auto instruction : block)
	{
		assert(instruction->isMachineInstruction());
	
		auto rule = _translationRules.find(instruction->opcodeString());
		
		// don't translate instructions with missing rules
		if(rule == _translationRules.end())
		{
			newInstructions.push_back(instruction->clone());
			
			continue;
		}
	
		auto instructions = rule->second->translateMachineInstruction(
			static_cast<machine::Instruction*>(instruction));

		// parallel-gather		
		newInstructions.insert(newInstructions.end(), instructions.begin(),
			instructions.end());
	}
	
	block.assign(newInstructions.begin(), newInstructions.end());
}

Pass* MachineToVIRInstructionTranslationPass::clone() const
{
	return new MachineToVIRInstructionTranslationPass(*this);
}

}

}

