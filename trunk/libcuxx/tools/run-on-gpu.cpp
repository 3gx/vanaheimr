/*! \file   run-on-gpu.cpp
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Wednesday November 13, 2013
	\brief  A command line utility for running standalone GPU applications.
*/

// GPU Native Includes
#include <gpunative/runtime/interface/Loader.h>
#include <gpunative/util/interface/ArgumentParser.h>

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
	util::ArgumentParser parser(argc, argv);
	
	parser.description("A direct loader for GPU binaries, accepts PTX and "
		"CUBIN formats.");
	
	std::string inputBinary;
	bool verbose = false;
	
	parser.parse("-i", "--input", inputBinary, "", "The path to the binary "
		"being executed (.ptx/.cubin).");
	parser.parse("-v", "--verbose", verbose, false,
		"Print out status information while running.");
	
	if(verbose)
	{
		util::enableAllLogs();
	}
	
	runBinary(inputBinary);
	
	return 0;
}


