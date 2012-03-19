/*! \file   ptx-to-vir-translator.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Monday February 13, 2012
	\brief  The source file for the ptx-to-vir-translator tool.
*/

// Vanaheimr Includes
#include <vanaheimr/ir/interface/Module.h>

#include <vanaheimr/compiler/interface/Compiler.h>

#include <vanaheimr/translation/interface/PTXToVIRTranslator.h>

// Ocelot Includes
#include <ocelot/ir/interface/Module.h>

// Hydrazine Includes
#include <hydrazine/implementation/ArgumentParser.h>

// Standard Library Includes
#include <fstream>

namespace vanaheimr
{

/*! \brief Load a PTX module, translate it to VIR, output the result */
static void translate(const std::string& virFileName,
	const std::string& ptxFileName)
{
	// Load the PTX module
	::ir::Module ptxModule(ptxFileName);
	
	compiler::Compiler* virCompiler = compiler::Compiler::getSingleton();
	
	// Translate the PTX
	translation::PTXToVIRTranslator translator(virCompiler);
	
	try
	{
		translator.translate(ptxModule);
	}
	catch(const std::exception& e)
	{
		std::cerr << "Compilation Failed: PTX to VIR translation failed.\n"; 
		std::cerr << "  Message: " << e.what() << "\n"; 
		return;
	}
	
	// Output the VIR module
	vanaheimr::compiler::Compiler::module_iterator virModule =
		virCompiler->getModule(ptxFileName);
	assert(virModule != virCompiler->module_end());
	
	virModule->name = virFileName;
	
	std::ofstream virFile(virFileName);
	
	if(!virFile.is_open())
	{
		std::cerr << "Compilation Failed: could not open VIR file '"
			<< virFileName << "' for writing.\n"; 
		return;
	}
	
	try
	{
		virModule->writeAssembly(virFile);
	}
	catch(const std::exception& e)
	{
		std::cerr << "Compilation Failed: binary writing failed.\n"; 
		std::cerr << "  Message: " << e.what() << "\n"; 
	}
}

}

int main(int argc, char** argv)
{
	hydrazine::ArgumentParser parser(argc, argv);

	std::string ptxFileName;
	std::string virFileName;

	parser.description("This program compiles a PTX file into a VIR binary.");

	parser.parse("-i", "--input",  ptxFileName, "", "The input PTX file path.");
	parser.parse("-o", "--output", virFileName,
		"", "The output VIR file path.");

	parser.parse();
	
	vanaheimr::translate(virFileName, ptxFileName);

	return 0;
}


