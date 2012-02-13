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
		Type* t = 0, Linkage l = ExternalLinkage, Constant* c = 0);
	~Global();

public:
	bool hasInitializer() const;

	Constant*       intializer();
	const Constant* initializer() const;

public:
	void setInitializer(Constant* c);

public:
	Global(const Global&)            = delete;
	Global& operator=(const Global&) = delete;
	
private:
	Constant* _initializer; // owned by the global

};

}

}


