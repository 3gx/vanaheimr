/*! \file   Variable.h
	\date   Friday February 3, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the Variable class.
*/

#pragma once 

namespace vanaheimr
{

namespace ir
{

/*! \brief Anything that can be defined at global scope */
class Variable
{
public:
	enum Linkage
	{
		ExternalLinkage = 0,//! Externally visible function
		LinkOnceAnyLinkage, //! Keep one copy of function when linking (inline)
		LinkOnceODRLinkage, //! Same, but only replaced by something equivalent.
		WeakAnyLinkage,     //! Keep one copy of named function when linking (weak)
		InternalLinkage,    //! Rename collisions when linking (static functions).
		PrivateLinkage      //! Like Internal, but omit from symbol table.
	};

public:
	Variable(const std::string& name, Module* module, Linkage linkage);

public:
	void setModule(Module* m);

public:
	const std::string& name() const;
	Module*            module();
	Linkage            linkage() const;

private:
	std::string _name;
	Module*     _module;
	Linkage     _linkage;

};

}

}

