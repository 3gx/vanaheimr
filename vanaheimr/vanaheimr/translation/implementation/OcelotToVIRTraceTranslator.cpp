/*!	\file   OcelotToVIRTraceTranslator.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Wednesday December 12, 2012
	\brief  The source file for the OcelotToVIRTraceTranslator class.
*/

// Vanaheimr Includes
#include <vanaheimr/translation/interface/OcelotToVIRTraceTranslator.h>
#include <vanaheimr/translation/interface/PTXToVIRTranslator.h>

#include <vanaheimr/compiler/interface/Compiler.h>

#include <vanaheimr/ir/interface/Type.h>

// Ocelot Includes
#include <ocelot/util/interface/ExtractedDeviceState.h>

#include <ocelot/ir/interface/Module.h>

// Standard Library Includes
#include <fstream>
#include <stdexcept>

namespace vanaheimr
{

namespace translation
{

OcelotToVIRTraceTranslator::OcelotToVIRTraceTranslator(compiler::Compiler* c)
: _compiler(c)
{

}

static void translatePTX(compiler::Compiler* compiler,
	const util::ExtractedDeviceState& state);
static void addVariablesForTraceData(compiler::Compiler* compiler,
	const util::ExtractedDeviceState& state);
static void archaeopteryxCodeGen(compiler::Compiler* compiler,
	const util::ExtractedDeviceState& state);

void OcelotToVIRTraceTranslator::translate(const std::string& traceFileName)
{
	std::ifstream stream(traceFileName.c_str());
	
	if(!stream.is_open())
	{
		throw std::runtime_error("Failed to open Ocelot trace file '" +
			traceFileName + "' for reading.\n");
	}
	
	util::ExtractedDeviceState state;
	
	state.deserialize(stream);
	
	translatePTX(_compiler, state);
	archaeopteryxCodeGen(_compiler, state);
	
	addVariablesForTraceData(_compiler, state);

	_translatedModuleName = state.launch.moduleName;
}

std::string OcelotToVIRTraceTranslator::translatedModuleName() const
{
	return _translatedModuleName;
}

static void translatePTX(compiler::Compiler* compiler,
	const util::ExtractedDeviceState& state)
{
	PTXToVIRTranslator ptxTranslator(compiler);
	
	auto module = state.modules.find(state.launch.moduleName);
	
	if(module == state.modules.end())
	{
		throw std::runtime_error("Malformed trace, no module names '" +
			state.launch.moduleName + "'.");
	}
	
	std::stringstream ptxStream(module->second->ptx);
	
	::ir::Module ptxModule(ptxStream, state.launch.moduleName);
	
	ptxTranslator.translate(ptxModule);
}

static void addTextures(compiler::Compiler* compiler,
	const util::ExtractedDeviceState& state)
{
	auto module = state.modules.find(state.launch.moduleName);
	
	if(module == state.modules.end())
	{
		throw std::runtime_error("Malformed trace, no module names '" +
			state.launch.moduleName + "'.");
	}
	
	assertM(module->second->textures.empty(),
		"Traces with textures not supported.");
}

static const ir::Type* getStringType(compiler::Compiler* compiler,
	const std::string& string)
{
	ir::ArrayType arrayType(compiler, compiler->getType("i8"),
		string.size() + 1);

	return *compiler->getOrInsertType(arrayType);
}

static void addGlobal(compiler::Compiler* compiler,
	const util::ExtractedDeviceState& state,
	const std::string& globalName, const std::string& globalValue)
{
	auto module = compiler->getModule(state.launch.moduleName);
	assert(module != compiler->module_end());
	
	auto global = module->newGlobal(globalName,
		getStringType(compiler, globalValue),
		ir::Global::ExternalLinkage,  ir::Global::Shared);

	auto constant = new ir::ArrayConstant(globalValue.c_str(),
		globalValue.size() + 1);
	
	global->setInitializer(constant);
}

static void addGlobal(compiler::Compiler* compiler,
	const util::ExtractedDeviceState& state,
	const std::string& globalName, const ::util::ByteVector& globalValue)
{
	auto module = compiler->getModule(state.launch.moduleName);
	assert(module != compiler->module_end());
	
	auto global = module->newGlobal(globalName,
		*compiler->getOrInsertType(ir::ArrayType(compiler,
			compiler->getType("i8"), globalValue.size())),
		ir::Global::ExternalLinkage, ir::Global::Shared);

	auto constant = new ir::ArrayConstant(globalValue.data(),
		globalValue.size());
	
	global->setInitializer(constant);
}

template <typename T>
static void addGlobal(compiler::Compiler* compiler,
	const util::ExtractedDeviceState& state,
	const std::string& globalName, const T& globalValue)
{
	std::stringstream stream;
	
	stream << globalValue;
	
	addGlobal(compiler, state, globalName, stream.str());
}

static void addLaunch(compiler::Compiler* compiler,
	const util::ExtractedDeviceState& state)
{
	if(state.launch.gridDim.y > 1 || state.launch.gridDim.z > 1)
	{
		throw std::runtime_error("Malformed trace, only support"
			" kernels with ctas in the x dimension.");
	}
	
	if(state.launch.blockDim.y > 1 || state.launch.blockDim.z > 1)
	{
		throw std::runtime_error("Malformed trace, only support"
			" kernels with threads in the x dimension.");
	}

	addGlobal(compiler, state, "simulated-parameter-memory-size",
		state.launch.parameterMemory.size());
	addGlobal(compiler, state, "simulated-parameter-memory",
		state.launch.parameterMemory);
	
	addGlobal(compiler, state, "simulated-ctas",
		state.launch.gridDim.x);
	addGlobal(compiler, state, "simulated-threads-per-cta",
		state.launch.blockDim.x);
	addGlobal(compiler, state, "simulated-shared-memory-per-cta",
		state.launch.sharedMemorySize);
	
	addGlobal(compiler, state, "simulated-kernel-name",
		state.launch.kernelName);
}

static void addAllocations(compiler::Compiler* compiler,
	const util::ExtractedDeviceState& state)
{
	if(!state.globalVariables.empty())
	{
		throw std::runtime_error("No support for global variables yet.");
	}
	
	for(auto allocation = state.globalAllocations.begin();
		allocation != state.globalAllocations.end(); ++allocation)
	{
		std::stringstream name;
		
		name << "simulated-allocation-" << allocation->second->devicePointer;
	
		addGlobal(compiler, state, name.str(), allocation->second->data);
	}
}

static void addAllocationChecks(compiler::Compiler* compiler,
	const util::ExtractedDeviceState& state)
{
	if(!state.postLaunchGlobalVariables.empty())
	{
		throw std::runtime_error("No support for global variables yet.");
	}

	for(auto allocation = state.postLaunchGlobalAllocations.begin();
		allocation != state.postLaunchGlobalAllocations.end(); ++allocation)
	{
		std::stringstream name;
		
		name << "simulated-verify-allocation-"
			<< allocation->second->devicePointer;
	
		addGlobal(compiler, state, name.str(), allocation->second->data);
	}
}

static void addVariablesForTraceData(compiler::Compiler* compiler,
	const util::ExtractedDeviceState& state)
{
	addTextures(compiler, state);
	addLaunch(compiler, state);
	addAllocations(compiler, state);
	addAllocationChecks(compiler, state);
}

static void archaeopteryxCodeGen(compiler::Compiler* compiler,
	const util::ExtractedDeviceState& state)
{
	codegen::ArchaeopteryxTarget target;

	for(auto module = compiler->begin(); module != compiler->end(); ++module)
	{
		target.assignModule(&*module);
		
		target.lower();
	}
}

}

}

