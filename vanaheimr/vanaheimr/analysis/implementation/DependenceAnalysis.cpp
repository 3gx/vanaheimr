/*! \file   DependenceAnalysis.cpp
	\date   Tuesday January 8, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the dependence analysis class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/analysis/interface/DependenceAnalysis.h>

namespace vanaheimr
{

namespace analysis
{

DependenceAnalysis::DependenceAnalysis()
: FunctionAnalysis("DependenceAnalysis", {"ControlFlowGraph"})
{

}

bool DependenceAnalysis::hasLocalDependence(const Instruction& predecessor,
	const Instruction& successor) const
{
	auto predecessors = getLocalPredecessors(successor);
	
	return predecessors.count(&predecessor) != 0;
}

bool DependenceAnalysis::hasDependence(const Instruction& predecessor,
	const Instruction& successor) const
{
	assertM(false, "not implemented");
}

DependenceAnalysis::InstructionSet DependenceAnalysis::getLocalPredecessors(
	const Instruction& successor)
{
	auto block = _predecessors.find(successor.block->id());
	
	if(block == _predecessors.end()) return InstructionSet();
	
	assert(sucessor.id() < block->second.size());
	
	return block->second[successor.id()];
}

DependenceAnalysis::InstructionSet DependenceAnalysis::getLocalSuccessors(
	const Instruction& predecessor)
{
	auto block = _successors.find(predecessor.block->id());
	
	if(block == _successors.end()) return InstructionSet();
	
	assert(sucessor.id() < block->second.size());
	
	return block->second[predecessor.id()];
}

void DependenceAnalysis::analyze(Function& function)
{
	// for all
	for(auto block = function.begin(); block != function.end(); ++block)
	{
		_setLocalDependences(*block);
	}
}

typedef DependenceAnalysis::InstructionSet InstructionSet;

static void addPredecessors(InstructionSet& predecessors,
	ir::BasicBlock::const_iterator instruction);

void DependenceAnalysis::_setLocalDependences(BasicBlock& block)
{
	auto predecessor = _predecessors.insert(std::make_pair(block.id(),
			InstructionSetVector())).first;
	auto successor   =   _successors.insert(std::make_pair(block.id(),
		InstructionSetVector())).first;
		
	predecessor->second.resize(block.size());
	  successor->second.resize(block.size());
	
	// TODO: do this with a prefix scan
	for(auto instruction : block)
	{
		InstructionSet& instructionPredecessors =
			predecessor->second[instruction.id()];
	
		_addPredecessors(instructionPredecessors, *instruction);
	}
	
	// TODO: collect successors in parallel
	for(auto instruction : block)
	{
		InstructionSet& instructionPredecessors =
			predecessor->second[instruction.id()];
		
		for(auto predecessor : instructionPredecessors)
		{
			InstructionSet& instructionSuccessors =
				successor->second[predecessor->id()];
		
			instructionSuccessors.insert(instruction);
		}
	}
}

static bool hasDataflowDependence(const ir::Instruction& predecessor,
	const ir::Instruction& successor)
{
	typedef util::SmallSet<ir::VirtualRegister*> VirtualRegisterSet;

	VirtualRegisterSet writes;

	for(auto write : predecessor.writes)
	{
		if(!write->isRegister()) continue;
	
		auto registerOperand = static_cast<ir::RegisterOperand*>(write);
	
		writes.insert(registerOperand->virtualRegister);
	}
	
	for(auto read : successor.reads)
	{
		if(!read->isRegister()) continue;
	
		auto registerOperand = static_cast<ir::RegisterOperand*>(read);
	
		if(writes.count(registerOperand->virtualRegister) != 0) return true;
	}
	
	return false;
}

static bool hasControlflowDependence(const ir::Instruction& predecessor,
	const ir::Instruction& successor)
{
	return predecessor.isBranch() || successor.isBranch();
}

static bool hasBarrierDependence(const ir::Instruction& predecessor,
	const ir::Instruction& successor)
{
	return predecessor.isBarrier() || successor.isBarrier();
}

static bool hasMemoryDependence(const ir::Instruction& predecessor,
	const ir::Instruction& successor)
{
	return (predecessor.accessesMemory() && successor.isStore()) ||
		(predecessor.isStore() && successor.accessesMemory());
}

static bool hasDependence(const ir::Instruction& predecessor,
	const ir::Instruction& successor)
{
	if(hasControlflowDependence(predecessor, successor)) return true;
	if(    hasBarrierDependence(predecessor, successor)) return true;
	if(     hasMemoryDependence(predecessor, successor)) return true;
	if(   hasDataflowDependence(predecessor, successor)) return true;
	
	return false;
}

static void addPredecessors(InstructionSet& predecessors,
	ir::BasicBlock::const_iterator instruction)
{
	auto end = instruction->block->rend();
	auto position = ir::BasicBlock::const_reverse_iterator(instruction);
	
	for(++position; position != end; ++position)
	{
		if(!hasDependence(*position, *instruction)) continue;
		
		predecessors.insert(position);
	}
}

}

}


