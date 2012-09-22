/*! \file   ConvertToSSAPass.cpp
	\date   Tuesday September 18, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the ConvertToSSAPass class.
*/

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/ConvertToSSAPass.h>

#include <vanaheimr/analysis/interface/DominatorAnalysis.h>
#include <vanaheimr/analysis/interface/DataflowAnalysis.h>
#include <vanaheimr/analysis/interface/ControlFlowGraph.h>

#include <vanaheimr/ir/interface/Function.h>
#include <vanaheimr/ir/interface/BasicBlock.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

namespace vanaheimr
{

namespace transforms
{

ConvertToSSAPass::ConvertToSSAPass()
: FunctionPass(StringVector({"DominanceAnalysis",
	"ControlFlowGraph", "DataflowAnalysis"}), "ConvertToSSAPass")
{

}

void ConvertToSSAPass::runOnFunction(Function& function)
{
	_insertPsis(function);
	_insertPhis(function);

	_rename(function);
}

void ConvertToSSAPass::_insertPhis(Function& function)
{
	typedef util::SmallSet<BasicBlock*> BasicBlockSet;

	// Insert Phis for live ins that are in the dominance frontier of
	//     any definition
	// 
	// Caveat: PHI placement may create additional definitions, so blocks in
	//         the DF of newly-placed PHIs need to be checked again
	
	// Get dependent analyses
	auto cfg = static_cast<ControlFlowGraph*>(getAnalysis("ControlFlowGraph"));
	auto dfg = static_cast<DataflowAnalysis*>(getAnalysis("DataflowAnalysis"));
	
	auto dominatorAnalysis = static_cast<DominatorAnalysis*>(
		getAnalysis("DominatorAnalysis"));
	
	// parallel for-all values
	for(auto value = function.register_begin();
		value != function.register_end(); ++value)
	{
		auto definingBlocks = getBlocksThatDefineThisValue(*value);

		BasicBlockSet blocksThatNeedPhis;
		
		if(!definingBlocks.empty())
		{
				// Do this with a local update, and then a final gather
				_registersNeedingRenaming.insert(&*value);
		}
		
		// The inner loop is sequential
		while(!definingBlocks.empty())
		{
			auto definingBlock = *definingBlocks.begin();
			definingBlocks.erase(definingBlocks.begin());
			
			auto dominanceFrontier = dominatorAnalysis->getDominanceFrontier(
				*definingBlock);
			
			for(auto frontierBlock : dominanceFrontier)
			{
				if(blocksThatNeedPhis.insert(frontierBlock).second)
				{
					definingBlocks.insert(frontierBlock);
				}
			}
		}
		
		// parallel version: sort by block and bulk-insert the phis
		for(block : blocksThatNeedsPhis)
		{
			_insertPhi(*value, *block);
		}
	}
}

void ConvertToSSAPass::_insertPsis(Function& function)
{
	// for all predicated instructions
	//  insert a PSI after conditional assignments
	
	// parallel for-all over blocks
	for(auto block = function.begin(); block != function.end(); ++block)
	{
		// posibly more parallelism in here over instructions,
		//  but phi insertions will require a scan
		for(auto instruction = block->begin();
			instruction != block->end(); ++instruction)
		{
			// skip not-predicated instructions
			if(instruction->guard->isAlwaysTrue()) continue;
			
			// skip instructions without outputs
			if(instruction->writes.empty()) continue;
			
			// add psis for all register writes
			auto next = instruction; ++next;
			
			for(auto write : instruction->writes)
			{
				assert(write->isRegister());
				
				auto registerWrite = static_cast<ir::RegisterOperand*>(write);
				
				auto psi = new ir::Psi(&block);
				
				psi->setGuard(ir::PredicateOperand::newTruePredicate(psi));
				psi->setD(new ir::RegisterOperand(
					registerWrite->virtualRegister, psi));
				
				block->insert(next, psi);
				
				// Do this with a local update, and then a final gather
				_registersNeedingRenaming.insert(
					registerWrite->virtualRegister);
			}
		}
	}
}

void ConvertToSSAPass::_insertPhi(VirtualRegister& vr, BasicBlock& block)
{
	auto phi = new ir::Phi(&block);
	
	phi->setD(new ir::RegisterOperand(&vr, phi));
	phi->setGuard(ir::PredicateOperand::newTruePredicate(phi));
				
	block.push_front(phi);
}

void ConvertToSSAPass::_rename(Function& f)
{
	BasicBlockSet worklist;
	
	 _renamedLiveIns.resize(f.size());
	_renamedLiveOuts.resize(f.size());
	
	// Start with blocks that defined renamed values
	// do this with a local insert, then a global gather + unique
	for(auto value : _registersNeedingRenaming)
	{
		auto definingBlocks = _getBlocksThatDefineThisValue(*value);
		
		_renameAllDefs(*value);
		
		worklist.insert(definingBlocks.begin(), definingBlocks.end());
	}
	
	while(!worklist.empty())
	{
		// update all blocks in the worklist in parallel
		_renameLocalBlocks(worklist);	
	}
	
	// delete all of the renamed registers
	for(auto value : _registersNeedingRenaming)
	{
		delete value;
	}
	
	_registersNeedingRenaming.clear();
	
	 _renamedLiveIns.clear();
	_renamedLiveOuts.clear();
}

void ConvertToSSAPass::_renameAllDefs(VirtualRegister& value)
{
	auto dfg = static_cast<DataflowAnalysis*>(getAnalysis("DataflowAnalysis"));
	auto dominatorAnalysis = static_cast<DominatorAnalysis*>(
		getAnalysis("DominatorAnalysis"));
	
	auto definitions = dfg->getReachingDefinitions(vr);
	
	for(definition : definitions)
	{
		// create a new value
		auto newValue = value.function->newVirtualRegister(value.type);
		
		// assign it to the def
		_updateDefinition(*definition, value, *newValue);
		
		// update uses in this block
		bool killed = _updateUsesInThisBlock(*definition, value, *newValue);
		
		// the value is not propagated further if it is killed in the same
		// block
		if(killed) continue;
		
		// add the mapping to the live in set of dominated blocks
		auto dominatedBlocks = dominatorAnalysis->getDominatedBlocks(
			*definition->block);
	
		for(dominatedBlock : dominatedBlocks)
		{
			VirtualRegisterMap& renamedLiveIns =
				_renamedLiveIns[dominatedBlock->id()];
			
			renamedLiveIns.insert(std::make_pair(&value, &*newValue));
		}
	}
}

void ConvertToSSAPass::_updateDefinition(Instruction& definingInstruction,
	VirtualRegister& value, VirtualRegister& newValue)
{
	for(auto write : instruction->writes)
	{
		assert(write->isRegister());
		
		auto writeOperand = static_cast<ir::RegisterOperand*>(write);
		
		if(&writeOperand->virtualRegister != &value) continue;
		
		// rename the register
		writeOperand->virtualRegister = &newValue;
	}
}

bool ConvertToSSAPass::_updateUsesInThisBlock(Instruction& definingInstruction,
	VirtualRegister& value, VirtualRegister& newValue)
{
	// scan forward over the block until the defining instruction is hit
	auto instruction = definingInstruction.block->begin();
	
	assert(instruction != definingInstruction.block->end());
		
	while(&*instruction != &definingInstruction)
	{
		++instruction;
		assert(instruction != definingInstruction.block->end());
	}
	
	// replace all uses in the block
	for(++instruction; instruction != definingInstruction.block->end();
		++instruction)
	{
		for(auto read : (*instruction)->reads)
		{
			// skip non-register reads
			if(!read->isRegister()) continue;
			
			auto readOperand = static_cast<ir::RegisterOperand*>(read);
			
			// skip reads from different registers
			if(readOperand->virtualRegister != &value) continue;
			
			// update the value
			readOperand->virtualRegister = renamedValue->second;
		}
		
		// stop on the first def
		for(auto write : instruction->writes)
		{
			assert(write->isRegister());
		
			auto writeOperand = static_cast<ir::RegisterOperand*>(write);
		
			// another value will be live out, not this one
			if(&writeOperand->virtualRegister == &value) return true;
		}
	}
	
	return false;
}

void ConvertToSSAPass::_renameLocalBlocks(BasicBlockSet& worklist)
{
	BasicBlockSet newList;

	// should be for-all
	for(auto block : worklist)
	{
		_renameValuesInBlock(newList, block);
	}

	// gather blocks to form the new worklist
	worklist = std::move(newList);
}

void ConvertToSSAPass::_renameValuesInBlock(
	BasicBlockSet& worklist, BasicBlock* block)
{
	VirtualRegisterMap& renamedLiveIns = _renamedLiveIns[block->id()];

	// replace all uses of values in this block, stop on the first def
	for(auto instruction : *block)
	{
		// replace reads
		for(auto read : instruction->reads)
		{
			// skip non-register reads
			if(!read->isRegister()) continue;
			
			auto readOperand = static_cast<ir::RegisterOperand*>(read);
			
			auto renamedValue = renamedLiveIns.find(
				readOperand->virtualRegister);
			
			// skip values that were not renamed
			if(renamedValue == renamedLiveIns.end()) continue;
		
			// update the value
			readOperand->virtualRegister = renamedValue->second;
		}
		
		// kill the update on writes
		for(auto write : instruction->writes)
		{
			assert(write->isRegister());
			
			auto writeOperand = static_cast<ir::RegisterOperand*>(write);
			
			auto renamedValue = renamedLiveIns.find(
				writeOperand->virtualRegister);
			
			// skip values that were not renamed
			if(renamedValue == renamedLiveIns.end()) continue;
			
			// kill renaming entries that are over-written
			renamedLiveIns.erase(renamedValue);
			
			// if all values were killed, don't queue up successors, just exit
			if(renamedLiveIns.empty()) return;
		}
	}
	
	// Any remaining renamed variables are live-out
	VirtualRegisterMap& renamedLiveOuts = _renamedLiveOuts[block->id()];

	renamedLiveOuts = std::move(renamedLiveIns);

	// add dominator tree successors with a renamed value as a live-in
	auto dominatorAnalysis = static_cast<DominatorAnalysis*>(
		getAnalysis("DominatorAnalysis"));
	
	auto dominatedBlocks = dominatorAnalysis->getDominatedBlocks(*block);
	
	for(auto dominatedBlock : dominatedBlocks)
	{
		auto dominatedBlockLiveIns = dfg->getLiveIns(*dominatedBlock);
	
		VirtualRegisterMap& dominatedBlockLiveInMap =
			_renamedLiveIns[dominatedBlock->id()];
	
		bool triggeredDominatedBlock = false;
	
		for(auto renamedValue : renamedLiveOuts)
		{
			if(dominatedBlockLiveIns.count(renamedValue) != 0)
			{
				triggeredDominatedBlock |= dominatedBlockLiveInMap.insert(
					*renamedValue).second;
			}
		}
		
		if(triggeredDominatedBlock)
		{
			worklist.insert(dominatedBlock);
		}
	}
}

ConverToSSAPass::BasicBlockSet ConvertToSSAPass:_getBlocksThatDefineThisValue(
	ir::VirtualRegister& value)
{
	auto dfg = static_cast<DataflowAnalysis*>(getAnalysis("DataflowAnalysis"));
	
	auto instructions = dfg->getReachingDefinitions(value);

	BasicBlockSet blocks;

	for(auto instruction : instructions)
	{
		blocks.insert(instruction->block);
	}

	return blocks;
}

}

}

