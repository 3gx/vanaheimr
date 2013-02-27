/*! \file   TranslationTableInstructionSelectionPass.cpp
	\date   Tuesday February 26, 2013
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the
		    TranslationTableInstructionSelectionPass class.
*/

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/TranslationTableInstructionSelectionPass.h>

namespace vanaheimr
{

namespace codegen
{

TranslationTableInstructionSelectionPass::TranslationTableInstructionSelectionPass()
: FunctionPass({}, "TranslationTableInstructionSelectionPass")
{
	
}

void TranslationTableInstructionSelectionPass::runOnFunction(Function& f)
{
	// TODO
}

}

}


