/*! \file   ArchaeopteryxABI.cpp
	\author Gregory Diamos <gdiamos@nvidia.com>
	\date   Friday December 21, 2021
	\brief  The source file for the ArchaeopteryxABI class.
*/

// Vanaheimr Includes
#include <vanaheimr/abi/interface/ArchaeopteryxABI.h>
#include <vanaheimr/abi/interface/ApplicationBinaryInterface.h>

#include <vanaheimr/ir/interface/Global.h>

namespace vanaheimr
{

namespace abi
{

ApplicationBinaryInterface* getArchaeopteryxABI()
{
	auto archaeopteryxABI = new ApplicationBinaryInterface;

	archaeopteryxABI->insert(new ApplicationBinaryInterface::FixedAddressRegion(
		"parameter", 1024, 8, ir::Global::Shared, 4096));

	return archaeopteryxABI;
}

}

}

