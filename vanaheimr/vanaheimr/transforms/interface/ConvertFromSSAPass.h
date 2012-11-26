/*! \file   ConvertFromSSAPass.h
	\date   Tuesday November 20, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the ConvertFromSSAPass class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/Pass.h>

namespace vanaheimr
{

namespace transforms
{

class ConvertFromSSAPass : public FunctionPass
{
public:
	ConvertFromSSAPass();

public:
	virtual void runOnFunction(Function& f);

private:
	void _removePhis(Function& f);
	void _removePsis(Function& f);

private:
	void _removePhis(ir::BasicBlock& block);

private:
	void removePsi(ir::Psi& psi);

};

}

}


