/*! \file   Loader.h
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Wednesday November 13, 2013
	\brief  The header file for the Loader class.
*/

#pragma once

// Standard Library Includes
#include <string>
#include <memory>
#include <vector>

namespace gpunative
{

namespace runtime
{

class LoaderState;

class Loader
{
public:
	typedef std::vector<std::string> StringVector;

public:
	/*! \brief Construct a new loader and associate it with a binary */
	Loader(const std::string& path, const StringVector& arguments, bool isEmbedded = false);
	~Loader();

public:
	Loader(const Loader&) = delete;
	Loader& operator=(const Loader&) = delete;

public:
	/*! \brief Load the associated binary. */
	void loadBinary();

	/*! \brief Run the associated binary. This assumes it has been loaded. */
	void runBinary();

	/*! \brief Get the return value */
	int getReturnValue() const;

private:
	std::string  _path;
	StringVector _arguments;
	bool         _isEmbedded;

	std::unique_ptr<LoaderState> _state;

};

}

}


