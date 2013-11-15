/*! \file   gpu-native-loader.cpp
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Wednesday November 13, 2013
	\brief  A command line utility for running standalone GPU applications.
*/

// GPU Native Includes
#include <gpu-native/runtime/interface/Loader.h>
#include <gpu-native/util/interface/ArgumentParser.h>

// Standard Library Includes
#include <string>
#include <stdexcept>

static void runBinary(const std::string& inputBinaryPath)
{
	gpunative::runtime::Loader loader(inputBinaryPath);
	
	try
	{
		loader.loadBinary();
	}
	catch(const std::exception& e)
	{
		std::cout << "Loading binary '" << inputBinaryPath << "' failed: "
			<< e.what() << "\n";
		return;
	}
	
	try
	{
		loader.runBinary();
	}
	catch(const std::exception& e)
	{
		std::cout << "Executing binary '" << inputBinaryPath << "' failed: "
			<< e.what() << "\n";
		return;
	}
}

int main(int argc, char** argv)
{
	gpunative::util::ArgumentParser parser(argc, argv);
	
	parser.description("A native loader for GPU binaries, accepts PTX and "
		"CUBIN formats.");
	
	std::string inputBinary;
	bool verbose = false;
	
	parser.parse("-i", "--input", inputBinary, "", "The path to the binary "
		"being executed (.ptx/.cubin).");
	parser.parse("-v", "--verbose", verbose, false,
		"Print out status information while running.");
	
	if(verbose)
	{
		gpunative::util::enableAllLogs();
	}
	
	runBinary(inputBinary);
	
	return 0;
}


