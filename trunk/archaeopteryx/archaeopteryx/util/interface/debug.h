/*! \file   debug.h
	\date   Sunday July 24, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for archaeopteryx debug functions.
*/

#pragma once

// Preprocessor macros
#ifdef device_assert
#undef device_assert
#endif
#define device_assert(x) _assert(x, "x", __FILE__, __LINE__)

namespace util
{

__device__ void _assert(bool condition, const char* expression,
	const char* filename, int line);

}

// TODO remove this when we get a linker
#include <archaeopteryx/util/implementation/debug.cpp>

