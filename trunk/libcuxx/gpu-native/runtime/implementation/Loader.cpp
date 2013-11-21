/*! \file   Loader.cpp
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Wednesday November 13, 2013
	\brief  The source file for the Loader class.
*/

// GPU Native Includes
#include <gpu-native/runtime/interface/Loader.h>

#include <gpu-native/driver/interface/CudaDriver.h>

#include <gpu-native/util/interface/Casts.h>
#include <gpu-native/util/interface/debug.h>

// Standard Library Includes
#include <fstream>
#include <stdexcept>
#include <cassert>

namespace gpunative
{

namespace runtime
{

typedef Loader::StringVector StringVector;

Loader::Loader(const std::string& path, const StringVector& arguments)
: _path(path), _arguments(arguments)
{

}

Loader::~Loader()
{

}

class LoaderState
{
public:
	LoaderState(const std::string& path, const StringVector& arguments);
	~LoaderState();
	
public:
	void runBinary();

private:
	typedef driver::CUcontext  CUcontext;
	typedef driver::CUmodule   CUmodule;
	typedef driver::CUfunction CUfunction;

private:
	void _loadState();
	int  _getDevice();
	void _runGlobalConstructors();
	void _runMain();

private:
	void _setupMainArguments();
	void _freeMainArguments();

private:
	std::string  _path;
	StringVector _arguments;
	CUcontext    _context;
	CUmodule     _module;
	CUfunction   _main;
	CUfunction   _init;
};

void Loader::loadBinary()
{
	// _state = std::make_unique<LoaderState>(_path, _arguments);

	_state = std::unique_ptr<LoaderState>(new LoaderState(_path, _arguments));
}

void Loader::runBinary()
{
	assert(_state);

	_state->runBinary();
}

LoaderState::LoaderState(const std::string& p, const StringVector& a)
: _path(p), _arguments(a), _context(0)
{
	_loadState();
}

LoaderState::~LoaderState()
{
	driver::CudaDriver::cuCtxDestroy(_context);
}

void LoaderState::runBinary()
{
	_runGlobalConstructors();
	_runMain();

	util::log("Loader") << "Successfully ran binary." << "\n";
}

static size_t getFileLength(std::istream& stream)
{
	size_t position = stream.tellg();
	
	stream.seekg(0, std::ios::end);
	
	size_t length = stream.tellg();

	stream.seekg(position);

	return length;
}

static std::string loadBinary(const std::string& path)
{
	util::log("Loader") << "Loading GPU binary from path: '" << path << "'\n";
	std::ifstream binaryFile(path);
	
	if(not binaryFile.is_open())
	{
		throw std::runtime_error("Could not open input binary file '" +
			path + "' for reading.");
	}
	
	size_t size = getFileLength(binaryFile);

	std::string result(size, ' ');
	
	binaryFile.read(const_cast<char*>(result.data()), size);
	
	util::log("Loader") << " loaded " << size << " bytes.\n";
	
	return result;
}

static void loadModule(driver::CUmodule& module, const std::string& binary)
{
	util::log("Loader") << "Loading module from binary data.\n";
	
	driver::CUjit_option options[] = {
	//	CU_JIT_TARGET,
		driver::CU_JIT_ERROR_LOG_BUFFER, 
		driver::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, 
	};

	const uint32_t errorLogSize       = 2048;
	uint32_t       errorLogActualSize = errorLogSize - 1;

	uint8_t errorLogBuffer[errorLogSize];

	std::memset(errorLogBuffer, 0, errorLogSize);

	void* optionValues[] = {
	//	(void*)CU_TARGET_COMPUTE_20,
		(void*)errorLogBuffer, 
		util::bit_cast<void*>(errorLogActualSize), 
	};

	try
	{
		driver::CudaDriver::cuModuleLoadDataEx(&module, binary.data(), 2,
			options, optionValues);
	}
	catch(const std::exception& e)
	{
		throw std::runtime_error("Failed to load binary data:\n\tMessage: " + std::string((char*)errorLogBuffer));
	}
	
}

void LoaderState::_loadState()
{
	util::log("Loader") << "Initializing CUDA driver.\n";
	
	driver::CudaDriver::cuInit(0);
	
	util::log("Loader") << "Creating context on devive " << _getDevice() << ".\n";
	
	driver::CudaDriver::cuCtxCreate(&_context, 0, _getDevice());
	
	auto binary = loadBinary(_path);
	
	
	loadModule(_module, binary);

	
	util::log("Loader") << "Loading 'main' function from module.\n";
	driver::CudaDriver::cuModuleGetFunction(&_main, _module, "_pre_main");
	
	util::log("Loader") << "Checking for global initialization function.\n";
	if(driver::CudaDriver::doesFunctionExist(_module, "__cxx_global_var_init"))
	{
		util::log("Loader") << "Loading '__cxx_global_var_init' function from module.\n";
		driver::CudaDriver::cuModuleGetFunction(&_init, _module,
			"__cxx_global_var_init");
	}
	else
	{
		_init = 0;
	}
}

int LoaderState::_getDevice()
{
	// TODO
	return 0;
}

void LoaderState::_runGlobalConstructors()
{
	// Skip programs without global constructors
	if(_init == 0) return;
	
	driver::CudaDriver::cuFuncSetBlockShape(_init, 1, 1, 1);
	driver::CudaDriver::cuFuncSetSharedSize(_init, 0);
	
	driver::CudaDriver::cuParamSetSize(_init, 0);
	driver::CudaDriver::cuLaunchGrid(_init, 1, 1);
}

void LoaderState::_runMain()
{
	util::log("Loader") << "Running 'main'.\n";

	// Setup dimensions
	util::log("Loader") << " setting up CTA dimensions (1, 1, 1).\n";
	driver::CudaDriver::cuFuncSetBlockShape(_main, 1, 1, 1);
	driver::CudaDriver::cuFuncSetSharedSize(_main, 0);
	
	// Set up argc and argv
	_setupMainArguments();
		
	// Allocate timers
	driver::CUevent start;
	driver::CUevent finish;
	
	driver::CudaDriver::cuEventCreate(&start,  0);
	driver::CudaDriver::cuEventCreate(&finish, 0);
	
	// Start timer
	driver::CudaDriver::cuEventRecord(start, 0);
	
	// Launch main
	util::log("Loader") << " launching kernel over grid (1, 1).\n";
	driver::CudaDriver::cuLaunchGrid(_main, 1, 1);

	// End the timer
	driver::CudaDriver::cuEventRecord(finish, 0);

	// Wait for the kernel
	util::log("Loader") << " waiting for kernel to finish....\n";
	driver::CudaDriver::cuEventSynchronize(finish);
	
	// Log the time
	float milliseconds = 0.0f;
	
	util::log("Loader") << " kernel finished in " << milliseconds << " ms\n";
	driver::CudaDriver::cuEventElapsedTime(&milliseconds, start, finish);
	
	// Destroy timers
	driver::CudaDriver::cuEventDestroy(start);
	driver::CudaDriver::cuEventDestroy(finish);

	// Free argc and argv
	_freeMainArguments();
	
}

void LoaderState::_setupMainArguments()
{
	util::log("Loader") << "Setting up arguments to main.\n";
	
	typedef std::vector<driver::CUdeviceptr> PointerVector;

	PointerVector argv;

	for(auto& argument : _arguments)
	{
		util::log("Loader") << " Registering memory for '" << argument << "'.\n";
		driver::CudaDriver::cuMemHostRegister(
			const_cast<char*>(argument.c_str()), 
			argument.size() + 1, driver::CU_MEMHOSTREGISTER_DEVICEMAP);
	
		driver::CUdeviceptr pointer = 0;
	
		driver::CudaDriver::cuMemHostGetDevicePointer(&pointer,
			const_cast<char*>(argument.c_str()), 0);
		util::log("Loader") << "  device pointer is '0x" << std::hex
			<< pointer << std::dec << "'.\n";
			
		argv.push_back(pointer);
	}
	
	size_t bytes = sizeof(int) + sizeof(driver::CUdeviceptr) * argv.size();
	util::log("Loader") << " setting parameter size to " << bytes << ".\n";
	
	driver::CudaDriver::cuParamSetSize(_main, bytes);
	
	size_t offset = 0;
	
	int argc = argv.size();
	
	util::log("Loader") << " setting up argc = " << argc << ".\n";
	driver::CudaDriver::cuParamSetv(_main, offset, &argc, 4);
	
	offset += sizeof(int);
	
	// TODO: Do we need alignment here?
	for(auto pointer : argv)
	{
		util::log("Loader") << " setting up argv[0x" << std::hex << (&pointer - &argv[0])
			<< std::dec << "] = 0x" << std::hex << pointer << std::dec << ".\n";
		driver::CudaDriver::cuParamSetv(_main, offset, &pointer,
			sizeof(driver::CUdeviceptr));
	
		offset += sizeof(driver::CUdeviceptr);
	}
}

void LoaderState::_freeMainArguments()
{
	for(auto& argument : _arguments)
	{
		driver::CudaDriver::cuMemHostUnregister(
			const_cast<char*>(argument.c_str()));
	}
}

}

}


