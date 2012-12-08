/*	\file   ArchaeopteryxDriver.cpp
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Friday December 7, 2012
	\brief  The source file for the ArchaeopteryxDriver class.
*/

// Archaeopteryx Includes
#include <archaeoteryx/driver/host-interface/ArchaeopteryDriver.h>
#include <archaeoteryx/util/host-interface/HostReflectionHost.h>

// Ocelot Includes
#include <ocelot/api/interface/ocelot.h>
#include <ocelot/cuda/interface/cuda_runtime.h>

// Standard Library Includes
const char ArchaeopteryxModule[] = {
	#include <ArchaeopteryxModule.inc>
};

namespace archaeopteryx
{

namespace driver
{

void ArchaeopteryxDriver::runSimulation(const std::string& traceFileName,
	const KnobList& knobs)
{
	_knobs = knobs;
	
	_loadTraceFile(traceFileName);
	
	_loadArchaeopteryxDeviceCode();
	
	_runSimulation();
	
	_unloadArchaeopteryxDeviceCode();
}

void ArchaeopteryxDriver::_loadArchaeopteryxDeviceCode()
{
	std::stringstream stream(ArchaeopteryxModule);
	ocelot::registerPTXModule(stream, "ArchaeopteryxModule");
	
	archaeopteryx::util::HostReflectionHost::create("ArchaeopteryxModule");
}

void ArchaeopteryxDriver::_runSimulation()
{
	
}

void ArchaeopteryxDriver::_unloadArchaeopteryxDeviceCode()
{
	archaeopteryx::util::HostReflectionHost::destroy();
	
	ocelot::unregisterModule("ArchaeopteryxModule");
}

}

}


