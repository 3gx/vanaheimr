/*! \file   File.h
	\date   Saturday April 23, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the File class.
*/

#pragma once

namespace util
{

/*! \brief Perform low level operations on a file from a CUDA kernel */
class File
{
public:
	/*! \brief Create a handle to a file */
	__device__ File(const char* fileName);

	/*! \brief Close the file */
	__device__ ~File();
	
public:
	/*! \brief Write data from a buffer into the file at the current offset */
	__device__ void write(const void* data, size_t size);

	/*! \brief Read data from the file at the current offset into a buffer */
	__device__ void read(void* data, size_t size);

	/*! \brief Get the size of the file */
	__device__ size_t size() const;
	
	/*! \brief Get the current get pointer */
	__device__ size_t tellg() const;
	
	/*! \brief Get the current put pointer */
	__device__ size_t tellp() const;
	
	/*! \brief Set the position of the get pointer */
	__device__ void seekg(size_t p);
	
	/*! \brief Set the position of the put pointer */
	__device__ void seekp(size_t p);
};

}

// TODO remove this when we get a real linker
#include <archaeopteryx/util/implementation/File.cpp>

