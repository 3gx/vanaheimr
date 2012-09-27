/*! \file   vir-optimizer.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Monday May 7, 2012
	\brief  The source file for the vir-optimizer tool.
*/

// Vanaheimr Includes
#include <vanaheimr/transforms/interface/PassManager.h>
#include <vanaheimr/transforms/interface/PassFactory.h>

#include <vanaheimr/asm/interface/BinaryReader.h>

#include <vanaheimr/ir/interface/Module.h>

// Hydrazine Includes
#include <hydrazine/interface/ArgumentParser.h>

// Standard Library Includes
#include <fstream>
#include <stdexcept>

namespace vanaheimr
{

static void optimizeModule(ir::Module* module, const std::string& optimizations)
{
	auto optimizationList = hydrazine::split(optimizations, ",");
	
	transforms::PassManager manager(module);
	
	for(auto optimization : optimizationList)
	{
		auto pass = transforms::PassFactory::createPass(optimization);

		if(pass == nullptr)
		{
			throw std::runtime_error("Failed to create pass named '"
				+ optimization + "'");
		}

		manager.addPass(pass);
	}
	
	manager.runOnModule();
}

static void optimize(const std::string& inputFileName,
	const std::string& outputFileName, const std::string& optimizations)
{	
	std::ios_base::openmode mode = std::ios_base::in | std::ios_base::binary;
	
	std::ifstream virFile(inputFileName.c_str(), mode);
	
	if(!virFile.is_open())
	{
		std::cerr << "ObjDump Failed: could not open VIR file '"
			<< inputFileName << "' for reading.\n"; 
		return;
	}
	
	ir::Module* module = 0;

	try
	{
		as::BinaryReader reader;

		module = reader.read(virFile, inputFileName);
	}
	catch(const std::exception& e)
	{
		std::cerr << "VIR Optimizer Failed: binary reading failed.\n"; 
		std::cerr << "  Message: " << e.what() << "\n"; 
		return;
	}
	
	try
	{
		optimizeModule(module, optimizations);
	}
	catch(const std::exception& e)
	{
		std::cerr << "VIR Optimizer Failed: optimization failed.\n"; 
		std::cerr << "  Message: " << e.what() << "\n"; 

		delete module;
		return;
	}
	
	std::ios_base::openmode oMode = std::ios_base::out | std::ios_base::binary;	
	
	std::ofstream outputVirFile(outputFileName.c_str(), oMode);
	
	if(!outputVirFile.is_open())
	{
		std::cerr << "ObjDump Failed: could not open VIR file '"
			<< outputFileName << "' for writing.\n"; 
		
		delete module;
		return;
	}
	
	try
	{
		module->writeBinary(outputVirFile);
	}
	catch(const std::exception& e)
	{
		std::cerr << "ObjDump Failed: binary writing failed.\n"; 
		std::cerr << "  Message: " << e.what() << "\n"; 
		delete module;
		return;
	}
	
	delete module;
}

}

int main(int argc, char** argv)
{
	hydrazine::ArgumentParser parser(argc, argv);

	std::string virFileName;
	std::string outputFileName;
	std::string optimizations;

	parser.description("This program reads in a VIR binary, optimizes it, "
		"and writes it out again a new binary.");

	parser.parse("-i", "--input" ,  virFileName,
		"", "The input VIR file path.");
	parser.parse("-o", "--output",  outputFileName,
		"", "The output VIR file path.");
	parser.parse("", "--optimizations",  optimizations,
		"", "Comma separated list of optimizations (ConvertToSSA).");
	parser.parse();
	
	vanaheimr::optimize(virFileName, outputFileName, optimizations);

	return 0;
}

