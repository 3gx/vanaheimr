/*! \file   StlFunctions.cpp
	\date   Sunday July 24, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for device equivalents of STL functions.
*/

// Archaeopteryx Includes
#include <archaeopteryx/util/interface/StlFunctions.h>

namespace util
{

template<typename Type>
__host__ __device__ Type min(Type a, Type b)
{
	return a < b ? a : b;
}

}


