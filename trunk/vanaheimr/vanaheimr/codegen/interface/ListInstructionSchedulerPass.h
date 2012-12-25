/*! \file   ListInstructionSchedulerPass.h
	\date   Sunday December 23, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the ListInstructionSchedulerPass class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/Pass.h>

namespace vanaheimr
{

namespace transforms
{

/*! \brief Convert a program IR not in SSA form to SSA */
class ListInstructionSchedulerPass : public FunctionPass
{
public:
	ListInstructionSchedulerPass();

public:
	void runOnFunction(Function& f);


};

}

}


