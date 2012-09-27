/*! \file   PassFactory.cpp
	\date   Wednesday May 2, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The source file for the PassFactory class.
	
*/

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/PassFactory.h>

#include <vanaheimr/transforms/interface/ConvertToSSAPass.h>

namespace vanaheimr
{

namespace transforms
{

Pass* PassFactory::createPass(const std::string& name,
	const StringVector& options)
{
	Pass* pass = nullptr;

	if(name == "ConvertToSSA" || name == "ConvertToSSAPass")
	{
		pass = new ConvertToSSAPass();
	}
	
	if(pass != nullptr)
	{
		pass->configure(options);
	}
	
	return pass;
}

}

}

