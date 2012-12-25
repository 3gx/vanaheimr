/*! \file   EnforceArchaeopteryxABIPass.h
	\author Gregory Diamos <gdiamos@nvidia.com>
	\date   Friday December 21, 2021
	\brief  The header file for the EnforceArchaeopteryxABIPass class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/Pass.h>

namespace vanaheimr
{

namespace codegen
{

/*! \brief A pass to enforce the archaeopteryx memory layout and
	calling convention */
class EnforceArchaeopteryxABIPass : public ModulePass
{
public:
	/*! \brief The constructor sets the type */
	EnforceArchaeopteryxABIPass();
	
public:
	/*! \brief Run the pass on a specific module */
	void runOnModule(Module& m);

};

}

}

