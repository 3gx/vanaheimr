/*! \file   debug.h
	\date   Sunday July 24, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for archaeopteryx debug functions.
*/

#pragma once

// Standard Library Includes
#include <iostream>

// Preprocessor macros
#ifdef device_assert
#undef device_assert
#endif
#define device_assert(x) _assert(x, "x", __FILE__, __LINE__)

#ifndef NDEBUG
	#define report(y) \
		if(REPORT_BASE > 0)\
		{ \
			{\
			std::cout << __FILE__ << ":"  << __LINE__  \
					<< ": " << y << "\n";\
			}\
		 \
		}
#else
	#define report(y)
#endif


namespace util
{

__device__ void _assert(bool condition, const char* expression,
	const char* filename, int line);

}

// TODO remove this when we get a linker
#include <archaeopteryx/util/implementation/debug.cpp>

