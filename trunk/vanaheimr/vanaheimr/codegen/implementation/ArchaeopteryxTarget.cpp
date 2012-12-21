/*! \file   ArchaeopteryxTarget.cpp
	\date   Thursday December 20, 2012
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The source file for the ArchaeopteryxTarget class.
*/

// Vanaheimr Includes
#include <vanaheimr/codegen/interface/ArchaeopteryxTarget.h>

#include <vanaheimr/transform/interface/PassManager.h>
#include <vanaheimr/transform/interface/PassFactory.h>


namespace vanaheimr
{

namespace codegen
{


ArchaeopteryxTarget::ArchaeopteryxTarget()
: Target("ArchaeopteryxSimulatorTarget"),
	registerAllocatorName("chaitan-briggs"), instructionSchedulerName("list")
{

}

void ArchaeopteryxTarget::lower()
{
	auto abiLowering = transform::PassFactory::createPass(
		"EnforceArchaeopetryxABIPass");

	if(abiLowering == nullptr)
	{
		throw std::runtime_error("Failed to create archaeopteryx"
			" ABI lowering pass.");
	}

	auto scheduler = transform::PassFactory::createPass(
		instructionSchedulerName);
	
	if(scheduler == nullptr)
	{
		delete abiLowering;

		throw std::runtime_error("Failed to get instruction scheduler '" +
			instructionSchedulerName +"'");
	}
	
	auto allocator = transform::PassFactory::createPass(registerAllocatorName);	
	
	if(allocator == nullptr)
	{
		delete abiLowering;
		delete scheduler;
	
		throw std::runtime_error("Failed to get register allocator '" +
			registerAllocatorName +"'");
	}
	
	transform::PassManager manager(_module);
	
	manager.addPass(abiLowering);
	manager.addPass(scheduler);
	manager.addPass(allocator);
	
	manager.addDependence(scheduler.name, abiLowering.name);
	manager.addDependence(allocator.name,   scheduler.name);
	
	manager.runOnModule();
}

ir::ModuleBase* ArchaeopteryxTarget::getLoweredModule()
{
	return _module;
}

}

}


