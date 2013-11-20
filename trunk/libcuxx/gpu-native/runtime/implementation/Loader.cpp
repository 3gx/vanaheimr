/*! \file   Loader.cpp
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Wednesday November 13, 2013
	\brief  The source file for the Loader class.
*/

// GPU Native Includes
#include <gpu-native/runtime/interface/Loader.h>
#include <gpu-native/driver/interface/CudaDriver.h>

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
	std::ifstream binaryFile(path);
	
	if(not binaryFile.is_open())
	{
		throw std::runtime_error("Could not open input binary file '" +
			path + "' for reading.");
	}
	
	std::string result;
	
	binaryFile.read(const_cast<char*>(result.data()),
		getFileLength(binaryFile));
	
	return result;
}

void LoaderState::_loadState()
{
	driver::CudaDriver::cuInit(0);
	
	driver::CudaDriver::cuCtxCreate(&_context, 0, _getDevice());
	
	auto binary = loadBinary(_path);
	
	driver::CudaDriver::cuModuleLoadDataEx(&_module, binary.data(), 0,
		nullptr, nullptr);
	driver::CudaDriver::cuModuleGetFunction(&_main, _module, "main");
	
	if(driver::CudaDriver::doesFunctionExist(_module, "__cxx_global_var_init"))
	{
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
	// Setup dimensions
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
	driver::CudaDriver::cuLaunchGrid(_main, 1, 1);

	// End the timer
	driver::CudaDriver::cuEventRecord(finish, 0);

	// Wait for the kernel
	driver::CudaDriver::cuEventSynchronize(finish);
	
	// Log the time
	float microseconds = 0.0f;
	
	driver::CudaDriver::cuEventElapsedTime(&microseconds, start, finish);
	
	// Destroy timers
	driver::CudaDriver::cuEventDestroy(start);
	driver::CudaDriver::cuEventDestroy(finish);
	
}

void LoaderState::_setupMainArguments()
{
	typedef std::vector<driver::CUdeviceptr> PointerVector;

	PointerVector argv;

	for(auto& argument : _arguments)
	{
		driver::CudaDriver::cuMemHostRegister(
			const_cast<char*>(argument.c_str()), 
			argument.size() + 1, driver::CU_MEMHOSTREGISTER_DEVICEMAP);
	
		driver::CUdeviceptr pointer = 0;
	
		driver::CudaDriver::cuMemHostGetDevicePointer(&pointer,
			const_cast<char*>(argument.c_str()), 0);
			
		argv.push_back(pointer);
	}
	
	size_t bytes = sizeof(int) + sizeof(driver::CUdeviceptr) * argv.size();
	
	driver::CudaDriver::cuParamSetSize(_main, bytes);
	
	size_t offset = 0;
	
	int argc = argv.size();
	
	driver::CudaDriver::cuParamSetv(_main, offset, &argc, sizeof(int));
	
	offset += sizeof(int);
	
	// TODO: Do we need alignment here?
	for(auto pointer : argv)
	{
		driver::CudaDriver::cuParamSetv(_main, offset, &pointer,
			sizeof(driver::CUdeviceptr));
	
		offset += sizeof(driver::CUdeviceptr);
	}
}

}

}


