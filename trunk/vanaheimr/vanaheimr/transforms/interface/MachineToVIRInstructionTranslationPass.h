/*! \file   MachineToVIRInstructionTranslationPass.h
	\date   Tuesday May 6, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the MachineToVIRInstructionTranslationPass
		    class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/Pass.h>

// Standard Library Includes
#include <vector>
#include <map>

// Forward Declarations
namespace vanaheimr { namespace ir { class Instruction; } }

namespace vanaheimr { namespace machine { class Instruction; } }

namespace vanaheimr
{

namespace transforms
{

/*! \brief Convert a program in the Machine IR into Vanaheimr IR */
class MachineToVIRInstructionTranslationPass : public BasicBlockPass
{
public:
	class TranslationRule
	{
	public:
		typedef std::vector<ir::Instruction*> InstructionVector;
	
	public:
		TranslationRule(const std::string& opcodeName);
		virtual ~TranslationRule();
		
	public:
		virtual InstructionVector translateMachineInstruction(
			const machine::Instruction* instruction) = 0;
	
		virtual TranslationRule* clone() const = 0;
	
	public:
		std::string opcode; // The opcode that the rule applies to
	};

	typedef MachineToVIRInstructionTranslationPass self;

public:
	MachineToVIRInstructionTranslationPass();
	~MachineToVIRInstructionTranslationPass();

public:
	MachineToVIRInstructionTranslationPass(const self&);
	MachineToVIRInstructionTranslationPass& operator=(const self&);

public:
	/*! \brief Add a translation rule to the pass, the pass will clone it */
	void addTranslationRule(const TranslationRule*);

	/*! \brief Clear any existing translation rules */
	void clearTranslationRules();

public:
	virtual void runOnBlock(BasicBlock& b);

public:
	virtual Pass* clone() const;

private:
	typedef std::map<std::string, TranslationRule*> OpcodeToRuleMap;
	
private:
	OpcodeToRuleMap _translationRules;

};

typedef MachineToVIRInstructionTranslationPass::TranslationRule
	MachineToVIRInstructionTranslationRule;

}

}

