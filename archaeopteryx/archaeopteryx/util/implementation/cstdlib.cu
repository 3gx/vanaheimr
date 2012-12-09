/*! \file   cstdlib.cu
	\date   Sunday December 9, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for device implementation of
			C standard library functions.
*/

#include <archaeopteryx/util/interface/cstdlib.h>

namespace archaeopteryx
{

namespace util
{

__device__ int atoi(const char* s)
{
	int value = 0;

	while(*s != '\n')
	{
		value = value * 10;
		
		value += *s - '0';
	}

	return value;
}

}

}


