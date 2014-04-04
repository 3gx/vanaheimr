/*! \file   GpuNativeMain.cpp
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Thursday April 3, 2014
	\brief  The source file for the GpuNativeMain interface.
*/

// Gpu Native Includes
#include <gpu-native/entry/interface/GpuNativeMain.h>
#include <gpu-native/runtime/interface/Loader.h>
#include <gpu-native/util/interface/debug.h>

// Standard Library Includes
#include <iostream>

static const char* ptx = nullptr;

extern void setEmbeddedPTX(const char* p)
{
	ptx = p;
}

extern const char* getEmbeddedPTX()
{
	return ptx;
}

extern int gpuNativeMain(int argc, const char** argv)
{
	#ifdef NDEBUG
	bool verbose = false;
	#else
	bool verbose = true;
	#endif

	if(verbose)
	{
		gpunative::util::enableAllLogs();
	}
	
	gpunative::runtime::Loader::StringVector arguments;
	
	for(int i = 0; i < argc; ++i)
	{
		arguments.push_back(argv[i]);
	}

	std::string inputBinaryPath = argv[0];
	
	gpunative::runtime::Loader loader(inputBinaryPath, arguments, true);

	try
	{
		loader.loadBinary();
	}
	catch(const std::exception& e)
	{
		std::cout << "Loading GPU binary '" << inputBinaryPath << "' failed: "
			<< e.what() << "\n";
		return -1;
	}
	
	try
	{
		loader.runBinary();
	}
	catch(const std::exception& e)
	{
		std::cout << "Executing GPU binary '" << inputBinaryPath << "' failed: "
			<< e.what() << "\n";
		return -2;
	}
	
	return loader.getReturnValue();
}

