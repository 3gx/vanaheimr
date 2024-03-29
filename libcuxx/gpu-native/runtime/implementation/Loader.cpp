/*! \file   Loader.cpp
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Wednesday November 13, 2013
	\brief  The source file for the Loader class.
*/

// GPU Native Includes
#include <gpu-native/runtime/interface/Loader.h>

#include <gpu-native/driver/interface/CudaDriver.h>

#include <gpu-native/entry/interface/GpuNativeMain.h>

#include <gpu-native/util/interface/Casts.h>
#include <gpu-native/util/interface/debug.h>
#include <gpu-native/util/interface/string.h>

// Standard Library Includes
#include <fstream>
#include <stdexcept>
#include <cassert>

namespace gpunative
{

namespace runtime
{

typedef Loader::StringVector StringVector;

Loader::Loader(const std::string& path, const StringVector& arguments, bool isEmbedded)
: _path(path), _arguments(arguments), _isEmbedded(isEmbedded)
{

}

Loader::~Loader()
{

}

class LoaderState
{
public:
	LoaderState(const std::string& path, const StringVector& arguments, bool isEmbedded);
	~LoaderState();
	
public:
	void runBinary();

public:
	int getReturnValue() const;

private:
	typedef driver::CUcontext  CUcontext;
	typedef driver::CUmodule   CUmodule;
	typedef driver::CUfunction CUfunction;

private:
	void _loadState();
	void _runGlobalConstructors();
	void _runMain();

private:
	void _setupMainArguments();
	void _freeMainArguments();

private:
	int  _getDevice();
	std::string _getDeviceName();

private:
	std::string  _path;
	StringVector _arguments;
	bool         _isEmbedded;

	CUcontext    _context;
	CUmodule     _module;
	CUfunction   _main;
	CUfunction   _init;

private:
	typedef std::vector<driver::CUdeviceptr> PointerVector;

private:
	PointerVector       _argv;
	driver::CUdeviceptr _argvPointer;

	int                 _returnValue;
	driver::CUdeviceptr _returnValuePointer;
};

void Loader::loadBinary()
{
	// _state = std::make_unique<LoaderState>(_path, _arguments);

	_state = std::unique_ptr<LoaderState>(new LoaderState(_path, _arguments, _isEmbedded));
}

void Loader::runBinary()
{
	assert(_state);

	_state->runBinary();
}

int Loader::getReturnValue() const
{
	assert(_state);
	
	return _state->getReturnValue();
}

LoaderState::LoaderState(const std::string& p, const StringVector& a, bool isEmbedded)
: _path(p), _arguments(a), _isEmbedded(isEmbedded), _context(0), _returnValue(-1)
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

	util::log("Loader") << "Successfully ran binary, exiting...." << "\n";
}

int LoaderState::getReturnValue() const
{
	return _returnValue;
}

static size_t getFileLength(std::istream& stream)
{
	size_t position = stream.tellg();
	
	stream.seekg(0, std::ios::end);
	
	size_t length = stream.tellg();

	stream.seekg(position);

	return length;
}

static bool isPTX(const std::string& binary)
{
	return binary.find(".version") != std::string::npos;
}

static std::string getName(const std::string& binary,
	size_t functionStart, size_t openBrace)
{
	if(functionStart == std::string::npos) return "";
	if(openBrace     == std::string::npos) return "";


	auto substring = binary.substr(functionStart, openBrace - functionStart);

	return util::removeWhitespace(substring);
}

static bool hasMain(const std::string& binary)
{
	util::log("Loader") << " searching for main:\n";
	size_t nextFunction = 0;

	while(nextFunction != std::string::npos)
	{
		nextFunction = binary.find(".func", nextFunction);

		if(nextFunction != std::string::npos)
		{
			nextFunction += 5;
		}

		// get the function name
		size_t openBrace = binary.find("(", nextFunction);
		
		auto name = getName(binary, nextFunction, openBrace);
		
		// handle the return argument list
		if(name.empty())
		{
			size_t argumentStart = binary.find(")", openBrace + 1);

			if(argumentStart != std::string::npos)
			{
				argumentStart += 1;
			}

			openBrace = binary.find("(", argumentStart);

			name = getName(binary, argumentStart, openBrace);
		}

		util::log("Loader") << "  checking function named '" << name << "'\n";
		if(name == "main")
		{
			return true;
		}

		if(nextFunction != std::string::npos)
		{
			nextFunction += 1;
		}
	}
	
	return false;
}

static bool hasPreMain(const std::string& binary)
{
	return binary.find(".entry _pre_main") != std::string::npos;
}

static void addPreMain(std::string& binary)
{
	char preMain[] = 
		"\n\n.visible .entry _pre_main(.param .b64 _retval, "
			".param .b64 _argv, .param .b32 _argc)\n"
		"{\n"
		"	.param .align 4 .b32 argc;\n"
		"	.param .align 4 .b64 argv;\n"
		"	.param .align 4 .b32 retval;\n"
		"	\n"
		"	.reg .s32 %r<3>;\n"
		"	.reg .s64 %l<3>;\n"
		"	\n"
		"	ld.param.u32 %r1,[_argc];\n"
		"	st.param.u32 [argc], %r1;\n"
		"	ld.param.u64 %l2,[_argv];\n"
		"	st.param.u64 [argv], %l2;\n"
		"	call (retval), main, (argc, argv);\n"
		"	ld.param.u32 %r2, [retval];\n"
		"	ld.param.u64 %l1, [_retval];\n" 	
		"	st.global.u32 [%l1], %r2;\n"
		"	\n"
		"	ret;\n"
		"}\n";

	binary.insert(binary.end(), preMain, preMain + sizeof(preMain) - 1);
}

static void patchStringConstants(std::string& binary)
{
	// TODO: Remove this when NVPTX BUG is fixed
	size_t position = 0;

	while(position != std::string::npos)
	{
		position = binary.find(".str", position);

		if(position != std::string::npos)
		{
			binary[position] = '_';

			position += 1;
		}
	}
}

static void patchWeak(std::string& binary)
{
	// TODO: Remove this when NVPTX bug is fixed
	size_t position = 0;

	while(position != std::string::npos)
	{
		position = binary.find(".weak", position);

		if(position != std::string::npos)
		{
			size_t endPosition = binary.find(".func", position);

			for(size_t i = position; i < endPosition; ++i)
			{
				binary[i] = ' ';
			}
			position += 5;

		}
	}
}

static void patchHidden(std::string& binary)
{
	// TODO: Remove this when NVPTX bug is fixed
	size_t position = 0;

	while(position != std::string::npos)
	{
		position = binary.find(".hidden", position);

		if(position != std::string::npos)
		{
			size_t endPosition = binary.find(".func", position);

			for(size_t i = position; i < endPosition; ++i)
			{
				binary[i] = ' ';
			}
			position += 7;

		}
	}
}

static void patchBinary(std::string& binary)
{
	if(!isPTX(binary)) return;

	patchStringConstants(binary);
	patchWeak(binary);
	patchHidden(binary);

	if(hasMain(binary))
	{
		if(!hasPreMain(binary))
		{
			addPreMain(binary);
		}
	}
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

	patchBinary(result);
	
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
		
		util::log("Loader") << "Binary is:" << binary << "\n";

		throw std::runtime_error("Failed to load binary data:\n\tMessage: " +
			std::string((char*)errorLogBuffer));
	}
	
}

static std::string getEmbeddedBinary()
{
	std::string result = getEmbeddedPTX();
	
	util::log("Loader") << " loaded " << result.size() << " bytes.\n";
	
	patchBinary(result);

	return result;
}

void LoaderState::_loadState()
{
	util::log("Loader") << "Initializing CUDA driver.\n";
	
	driver::CudaDriver::cuInit(0);
	
	util::log("Loader") << "Creating context on device " << _getDevice()
		<< ": name '" << _getDeviceName() << "'.\n";
	
	driver::CudaDriver::cuCtxCreate(&_context, 0, _getDevice());
	
	std::string binary;

	if(_isEmbedded)
	{
		binary = getEmbeddedBinary();
	}
	else
	{
		binary = loadBinary(_path);
	}

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
		util::log("Loader") << " No global initialization function found.\n";
		_init = 0;
	}
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
	
	driver::CudaDriver::cuEventElapsedTime(&milliseconds, start, finish);
	util::log("Loader") << " kernel finished in " << milliseconds << " ms\n";
	
	// Destroy timers
	driver::CudaDriver::cuEventDestroy(start);
	driver::CudaDriver::cuEventDestroy(finish);

	// Free argc and argv
	_freeMainArguments();

	util::log("Loader") << " kernel returned value " << _returnValue << "\n";
	
}

void LoaderState::_setupMainArguments()
{
	util::log("Loader") << " Setting up arguments to main.\n";
	
	// Register each of the argv entries
	_argv.clear();

	for(auto& argument : _arguments)
	{
		util::log("Loader") << "  Registering memory for '" << argument << "'.\n";
		driver::CudaDriver::cuMemHostRegister(
			const_cast<char*>(argument.c_str()), 
			argument.size() + 1, driver::CU_MEMHOSTREGISTER_DEVICEMAP);
	
		driver::CUdeviceptr pointer = 0;
	
		driver::CudaDriver::cuMemHostGetDevicePointer(&pointer,
			const_cast<char*>(argument.c_str()), 0);
		util::log("Loader") << "   device pointer is '0x" << std::hex
			<< pointer << std::dec << "', host pointer is '"
			<< (void*)argument.data() << "'.\n";
			
		_argv.push_back(pointer);
	}
	
	// Register the argv array
	driver::CudaDriver::cuMemHostRegister(
		_argv.data(), 
		_argv.size() * sizeof(driver::CUdeviceptr),
		driver::CU_MEMHOSTREGISTER_DEVICEMAP);

	_argvPointer = 0;
	driver::CudaDriver::cuMemHostGetDevicePointer(&_argvPointer,
		_argv.data(), 0);
	
	// Register the return value
	driver::CudaDriver::cuMemHostRegister(
		&_returnValue, sizeof(int),
		driver::CU_MEMHOSTREGISTER_DEVICEMAP);

	_returnValuePointer = 0;
	driver::CudaDriver::cuMemHostGetDevicePointer(&_returnValuePointer,
		&_returnValue, 0);

	// Set the parameters	
	// return value pointer (8 bytes)
	// argv pointer         (8 bytes)
	// argc value           (4 bytes)

	size_t bytes = sizeof(int) + sizeof(driver::CUdeviceptr) * 2;
	
	util::log("Loader") << "  setting parameter size to " << bytes << ".\n";
	
	driver::CudaDriver::cuParamSetSize(_main, bytes);
	
	util::log("Loader") << "  setting up return value pointer "
		<< " = 0x" << std::hex << _returnValuePointer << std::dec << " (offset "
		<< 0 << ", size " << sizeof(driver::CUdeviceptr) << ").\n";
	driver::CudaDriver::cuParamSetv(_main, 0, &_returnValuePointer, sizeof(driver::CUdeviceptr));
	util::log("Loader") << "  setting up argv pointer "
		<< " = 0x" << std::hex << _argvPointer << std::dec << " (offset "
		<< 0 << ", size " << sizeof(driver::CUdeviceptr) << ").\n";
	driver::CudaDriver::cuParamSetv(_main, 8, &_argvPointer, sizeof(driver::CUdeviceptr));

	int argc = _argv.size();
	
	util::log("Loader") << "  setting up argc = " << argc << ".\n";
	driver::CudaDriver::cuParamSetv(_main, 16, &argc, sizeof(int));
}

void LoaderState::_freeMainArguments()
{
	for(auto& argument : _arguments)
	{
		driver::CudaDriver::cuMemHostUnregister(
			const_cast<char*>(argument.c_str()));
	}
	
	driver::CudaDriver::cuMemHostUnregister(_argv.data());
	driver::CudaDriver::cuMemHostUnregister(&_returnValue);
}

int LoaderState::_getDevice()
{
	// TODO
	return 0;
}

std::string LoaderState::_getDeviceName()
{
	char name[2048];

	driver::CudaDriver::cuDeviceGetName(name, sizeof(name), _getDevice());
	
	return name;	
}

}

}


