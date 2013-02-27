/*! \file   TranslationTableEntry.cpp
	\date   Thursday February 23, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the TranslationTableEntry class.
*/

// Vanaheimr Includes
#include <vanaheimr/machine/interface/TranslationTableEntry.h>

namespace vanaheimr
{

namespace machine
{

TranslationTableEntry::Translation::Argument::Argument(Type t, unsigned int i)
: type(t), index(i)
{

}

TranslationTableEntry::Translation::Argument::Argument(const std::string& _imm)
: type(Immediate), immediate(_imm)
{
	
}

bool TranslationTableEntry::Translation::Argument::isTemporary() const
{
	return type == Temporary;
}

bool TranslationTableEntry::Translation::Argument::isImmediate() const
{
	return type == Immediate;
}

bool TranslationTableEntry::Translation::Argument::isRegister() const
{
	return type == Register || type == Address;
}

bool TranslationTableEntry::Translation::Argument::isAddress() const
{
	return type == Address;
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

	for(TranslationVector::const_iterator entry = translations.begin();
		entry != translations.end(); ++entry)
	{
		for(ArgumentVector::const_iterator argument = entry->arguments.begin();
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


