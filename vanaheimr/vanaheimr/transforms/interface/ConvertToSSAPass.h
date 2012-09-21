/*! \file   ConvertToSSAPass.h
	\date   Tuesday September 18, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the ConvertToSSAPass class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/Pass.h>

namespace vanaheimr
{

namespace transforms
{

/*! \brief Convert a program IR not in SSA form to SSA */
class ConvertToSSAPass : public FunctionPass
{
public:
	ConvertToSSAPass();

public:
	virtual void runOnFunction(Function& f);

private:
	typedef util::LargeSet<VirtualRegister*> VirtualRegisterSet;
	typedef util::LargeSet<BasicBlock*> BasicBlockSet;

private:
	void _insertPhis(Function& f);
	void _insertPsis(Function& f);
	
	void _insertPhi(VirtualRegister& vr, BasicBlock& block);

	void _renameAllDefs(VirtualRegister& vr);
	
	void _rename(Function& f);
	void _renameLocalBlocks(BasicBlockSet& worklist);
	void _renameValuesInBlock(BasicBlockSet& worklist, BasicBlock* block);

private:

private:

private:
	VirtualRegisterSet _registersNeedingRenaming;

};

}

}

