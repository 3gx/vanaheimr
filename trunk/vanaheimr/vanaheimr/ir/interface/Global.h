/*! \file   Global.h
	\date   Friday February 3, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Global class.
*/

#pragma once

// Vanaheimr Includes
#include <vanaheimr/ir/interface/BasicBlock.h>
#include <vanaheimr/ir/interface/Argument.h>

namespace vanaheimr
{

namespace ir
{

/*! \brief Describes a vanaheimr function */
class Global : public Variable
{
public:
	Global(const std::string& name = "", Module* m = 0,
		Linkage l = ExternalLinkage);
	
public:
	Constant* intializer();
	const Constant* initializer() const;

public:
	void setInitializer(Constant* c);

};

}

}


