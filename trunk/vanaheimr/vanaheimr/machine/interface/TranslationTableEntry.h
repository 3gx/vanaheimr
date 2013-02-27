/*! \file   TranslationTableEntry.h
	\date   Wednesday February 1, 2012
	\author Gregory Diamos <gdiamos@nvidia.com>
	\brief  The header file for the TranslationTableEntry class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <vector>

// Forward Declarations
namespace vanaheimr { namespace machine { class Operation; } }

namespace vanaheimr
{

namespace machine
{

/*! \brief A rule for translating a VIR operation into a Machine equivalent  */
class TranslationTableEntry
{
public:
	/*! \brief Describes how to translate a VIR instruction to a logical op */
	class Translation
	{
	public:
		class Argument
		{
		public:
			enum Type
			{
				Register  = 0,
				Temporary = 1,
				Immediate = 2,
				Address   = 3
			};

		public:
			Argument(Type _type, unsigned int index);
			Argument(const std::string& _imm);

		public:
			bool isTemporary() const;
			bool isImmediate() const;
			bool isRegister()  const;
			bool isAddress()   const;

		public:
			Type         type;
			std::string  immediate;
			unsigned int index;
		};

		typedef std::vector<Argument> ArgumentVector;

	public:
		Translation(const Operation* _lop);

	public:
		const Operation* operation;
		ArgumentVector   arguments;

	};

	typedef std::vector<Translation> TranslationVector;
	typedef Translation::ArgumentVector ArgumentVector;

	typedef TranslationVector::iterator       iterator;
	typedef TranslationVector::const_iterator const_iterator;

public:
	TranslationTableEntry(const std::string& _name);

public:
	unsigned int totalArguments() const;
	unsigned int totalTemporaries() const;

public:
	iterator       begin();
	const_iterator begin() const;

	iterator       end();
	const_iterator end() const;

public:
	size_t size()  const;
	bool   empty() const;

public:
	std::string       name;         // name of the VIR operation to translate
	TranslationVector translations; // translation into logical ops
};

}

}



