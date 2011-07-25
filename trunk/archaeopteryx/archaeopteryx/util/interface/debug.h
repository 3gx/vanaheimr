/*! \file   debug.h
	\date   Sunday July 24, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for archaeopteryx debug functions.
*/

#pragma once

// Preprocessor macros
#ifndef assert
#define assert(x) _assert(x, "x", __FILE__, __LINE__)
#endif

namespace util
{

__host__ __device__ void _assert(bool condition, const char* expression,
	const char* filename, int line);

}

// TODO remove this when we get a linker
#include <archaeopteryx/util/implementation/debug.cpp>

