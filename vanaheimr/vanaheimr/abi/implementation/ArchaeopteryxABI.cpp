/*! \file   ArchaeopteryxABI.cpp
	\author Gregory Diamos <gdiamos@nvidia.com>
	\date   Friday December 21, 2021
	\brief  The source file for the ArchaeopteryxABI class.
*/

// Vanaheimr Includes
#include <vanaheimr/abi/interface/ArchaeopteryxABI.h>
#include <vanaheimr/abi/interface/ApplicationBinaryInterface.h>

namespace vanaheimr
{

namespace abi
{

ApplicationBinaryInterface* getArchaeopteryxABI()
{
	auto archaeopteryxABI = new ApplicationBinaryInterface;

	// TODO initialize this

	return archaeopteryxABI;
}

}

}

