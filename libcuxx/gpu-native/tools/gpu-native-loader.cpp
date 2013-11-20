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

typedef gpunative::runtime::Loader::StringVector StringVector;

static void runBinary(const std::string& inputBinaryPath,
	const StringVector& arguments)
{
	gpunative::runtime::Loader loader(inputBinaryPath, arguments);
	
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

static StringVector getArguments(const std::string& path, int argc, char** argv)
{
	StringVector arguments;
	
	arguments.push_back(path);
	
	int position = 0;
	
	for( ; position < argc; ++position)
	{
		if(std::string(argv[position]) == "--")
		{
			++position;
			break;
		}
	}
	
	for( ; position < argc; ++position)
	{
		arguments.push_back(argv[position]);
	}
	
	return arguments;
}

int main(int argc, char** argv)
{
	gpunative::util::ArgumentParser parser(argc, argv);
	
	parser.description("A native loader for GPU binaries, accepts PTX and "
		"CUBIN formats.\n"
		"\tRun with: gpu-native-loader [options] -- [gpu-program-options]");
	
	std::string inputBinary;
	bool verbose = false;
	
	parser.parse("-i", "--input", inputBinary, "", "The path to the binary "
		"being executed (.ptx/.cubin).");
	parser.parse("-v", "--verbose", verbose, false,
		"Print out status information while running.");
	parser.parse();
	
	if(verbose)
	{
		gpunative::util::enableAllLogs();
	}
	
	runBinary(inputBinary, getArguments(inputBinary, argc, argv));
	
	return 0;
}


