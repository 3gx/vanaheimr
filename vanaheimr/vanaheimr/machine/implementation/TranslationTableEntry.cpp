/*! \file   TranslationTableEntry.cpp
	\date   Thursday February 23, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the TranslationTableEntry class.
*/

// Vanaheimr Includes
#include <vanaheimr/machine/interface/TranslationTableEntry.h>

#include <vanaheimr/ir/interface/Constant.h>

namespace vanaheimr
{

namespace machine
{

TranslationTableEntry::Translation::Argument::Argument(unsigned int i, bool s)
: immediate(nullptr), type(nullptr), index(i), isSource(s),
	_argumentType(Register)
{

}

TranslationTableEntry::Translation::Argument::Argument(unsigned int i,
 const Type* t)
: immediate(nullptr), type(t), index(i), isSource(false),
	_argumentType(Temporary)
{

}

TranslationTableEntry::Translation::Argument::Argument(const Constant* c)
: immediate(c), type(c->type()), index(0), isSource(false),
	_argumentType(Immediate)
{

}

bool TranslationTableEntry::Translation::Argument::isTemporary() const
{
	return _argumentType == Temporary;
}

bool TranslationTableEntry::Translation::Argument::isImmediate() const
{
	return _argumentType == Immediate;
}

bool TranslationTableEntry::Translation::Argument::isRegister() const
{
	return _argumentType == Register;
}

TranslationTableEntry::Translation::Translation(const Operation* l)
: operation(l)
{
	
}

TranslationTableEntry::TranslationTableEntry(const std::string& n)
: name(n)
{

}

unsigned int TranslationTableEntry::totalArguments() const
{
	unsigned int args = 0;

	for(TranslationVector::const_iterator entry = translations.begin();
		entry != translations.end(); ++entry)
	{
		for(ArgumentVector::const_iterator argument = entry->arguments.begin();
			argument != entry->arguments.end(); ++argument)
		{
			if(argument->isRegister())
			{
				args = std::max(args, argument->index + 1);
			}
		}
	}
	
	return args;
}

unsigned int TranslationTableEntry::totalTemporaries() const
{
	unsigned int temps = 0;

	for(auto entry = translations.begin();
		entry != translations.end(); ++entry)
	{
		for(auto argument = entry->arguments.begin();
			argument != entry->arguments.end(); ++argument)
		{
			if(argument->isTemporary())
			{
				temps = std::max(temps, argument->index + 1);
			}
		}
	}
	
	return temps;
}

TranslationTableEntry::ArgumentVector
	TranslationTableEntry::getTemporaries() const
{
	ArgumentVector temps(totalTemporaries(), Translation::Argument(nullptr));
	
	for(auto entry = translations.begin();
		entry != translations.end(); ++entry)
	{
		for(auto argument = entry->arguments.begin();
			argument != entry->arguments.end(); ++argument)
		{
			if(argument->isTemporary())
			{
				temps[argument->index] = *argument;
			}
		}
	}
	
	return temps;
}

TranslationTableEntry::iterator TranslationTableEntry::begin()
{
	return translations.begin();
}

TranslationTableEntry::const_iterator TranslationTableEntry::begin() const
{
	return translations.begin();
}

TranslationTableEntry::iterator TranslationTableEntry::end()
{
	return translations.end();
}

TranslationTableEntry::const_iterator TranslationTableEntry::end() const
{
	return translations.end();
}

size_t TranslationTableEntry::size() const
{
	return translations.size();
}

bool TranslationTableEntry::empty() const
{
	return translations.empty();
}

}

}


