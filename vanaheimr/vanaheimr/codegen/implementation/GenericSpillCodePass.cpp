/*! \file   GenericSpillCodePass.cpp
	\date   Wednesday January 2, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the GenericSpillCodePass class.
*/

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/GenericSpillCodePass.h>

// Hydrazine Includes
#include <hydrazine/interface/debug.h>

namespace vanaheimr
{

namespace codegen
{

GenericSpillCodePass::GenericSpillCodePass()
: FunctionPass({}, "GenericSpillCodePass")
{

}

void GenericSpillCodePass::runOnFunction(Function& f)
{
	assertM(false, "Not implemented.");
}

}

}


