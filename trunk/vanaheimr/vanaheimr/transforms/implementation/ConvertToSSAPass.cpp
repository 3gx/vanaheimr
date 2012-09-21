/*! \file   ConvertToSSAPass.cpp
	\date   Tuesday September 18, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the ConvertToSSAPass class.
*/

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/ConvertToSSAPass.h>

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
	
	// Start with blocks that defined renamed values
	// do this with a local insert, then a global gather + unique
	for(auto value : _registersNeedingRenaming)
	{
		auto definingBlocks = getBlocksThatDefineThisValue(*value);
		
		_renameAllDefs(*value);
		
		worklist.insert(definingBlocks.begin(), definingBlocks.end());
	}
	
	_registersNeedingRenaming.clear();
	
	while(!worklist.empty())
	{
		// update all blocks in the worklist in parallel
		_renameLocalBlocks(worklist);	
	}
}

void ConvertToSSAPass::_renameAllDefs(VirtualRegister& vr)
{
	// TODO
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

bool ConvertToSSAPass::_renameValuesInBlock(BasicBlock* block)
{
	// TODO
}

}

}

