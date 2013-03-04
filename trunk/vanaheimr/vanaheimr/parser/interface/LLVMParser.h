/*! \file   LLVMParser.h
	\date   March 3, 2013
	\author Gregory Diamos <solusstultus@gmail.com>
	\brief  The header file for the LLVM parser class.
*/

#pragma once

namespace vanaheimr
{

namespace parser
{

/*! \brief A parser for the low level virtual machine assembly language */
class LLVMParser
{
public:
	LLVMParser(compiler::Compiler* compiler);

public:
	void parse(const std::string& filename);

public:
	std::string getParsedModuleName() const;

private:
	compiler::Compiler* _compiler;
	std::string         _moduleName;

};

}


}



