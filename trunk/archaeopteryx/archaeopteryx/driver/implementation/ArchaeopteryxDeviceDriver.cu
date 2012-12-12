/*	\file   ArchaeopteryxDeviceDriver.cu
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Friday December 7, 2012
	\brief  The source file for the ArchaeopteryxDeviceDriver class.
*/

// Archaeopteryx Includes
#include <archaeopteryx/driver/interface/ArchaeopteryxDeviceDriver.h>
#include <archaeopteryx/driver/interface/SimulatorKnobs.h>

#include <archaeopteryx/runtime/interface/Runtime.h>

#include <archaeopteryx/util/interface/Knob.h>
#include <archaeopteryx/util/interface/debug.h>

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace archaeopteryx
{

namespace driver
{
	
__device__ ArchaeopteryxDeviceDriver::ArchaeopteryxDeviceDriver()
{
	util::KnobDatabase::create();
	rt::Runtime::create();
}

__device__ ArchaeopteryxDeviceDriver::~ArchaeopteryxDeviceDriver()
{
	rt::Runtime::destroy();
	util::KnobDatabase::destroy();
}

__device__ void ArchaeopteryxDeviceDriver::loadKnobs(
	const void* serializedKnobs)
{
	const char* base     = (const char*) serializedKnobs;
	const char* iterator = base;	

	const SimulatorKnobs* header = (const SimulatorKnobs*)iterator;
	iterator += sizeof(SimulatorKnobs);

	for(unsigned int knob = 0; knob != header->knobCount; ++knob)
	{
		const SimulatorKnobs::KnobOffsetPair* offsets = 
			(const SimulatorKnobs::KnobOffsetPair*) iterator;
		iterator += sizeof(SimulatorKnobs::KnobOffsetPair);

		const char* knobName  = base + offsets->first;
		const char* knobValue = base + offsets->second;

		device_report("Loaded knob (%s, %s)\n", knobName, knobValue);

		util::KnobDatabase::addKnob(new util::Knob(knobName, knobValue));
	}
}

__device__ void ArchaeopteryxDeviceDriver::runSimulation()
{
	_loadFile();
	_extractSimulatorParameters();
	_loadInitialMemoryContents();
	_runSimulation();
	_verifyMemoryContents();
}

__device__ void ArchaeopteryxDeviceDriver::_loadFile()
{
	util::string fileName =
		util::KnobDatabase::getKnob<util::string>("TraceFileName");

	rt::Runtime::loadBinary(fileName.c_str());
	
	device_report("loaded binary...\n");
}

__device__ static void addKnobFromBinary(ir::Binary* binary,
	const char* knobName)
{
	device_report(" Getting symbol (%s)\n", knobName);

	util::string value = binary->getSymbolDataAsString(knobName);
	
	device_report("  Loaded knob (%s, %s)\n", knobName, value.c_str());

	util::KnobDatabase::addKnob(new util::Knob(knobName, value));
}

__device__ void ArchaeopteryxDeviceDriver::_extractSimulatorParameters()
{
	device_report("Extracting simulator parameters from binary.\n");
	
	ir::Binary* binary = rt::Runtime::getSelectedBinary();

	addKnobFromBinary(binary, "simulator-ctas"                 );
	addKnobFromBinary(binary, "parameter-memory-size"          );
	addKnobFromBinary(binary, "simulator-threads-per-cta"      );
	addKnobFromBinary(binary, "simulator-shared-memory-per-cta");
}

__device__ void ArchaeopteryxDeviceDriver::_loadInitialMemoryContents()
{

}

__device__ void ArchaeopteryxDeviceDriver::_runSimulation()
{

}

__device__ void ArchaeopteryxDeviceDriver::_verifyMemoryContents()
{

}

}

}

extern "C" __global__ void archaeopteryxDriver(const void* knobs)
{
	archaeopteryx::driver::ArchaeopteryxDeviceDriver driver;

	driver.loadKnobs(knobs);
	driver.runSimulation();
}

