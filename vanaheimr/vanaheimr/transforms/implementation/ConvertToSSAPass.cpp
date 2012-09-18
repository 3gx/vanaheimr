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
: FunctionPass(StringVector(), "ConvertToSSAPass")
{

}

void ConvertToSSAPass::runOnFunction(Function& f)
{
	assertM(false, "Not implemented.");
}

}

}

