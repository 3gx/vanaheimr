/*! \file   StlFunctions.h
	\date   Sunday July 24, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for device equivalents of STL functions.
*/

#pragma once

namespace archaeopteryx
{

namespace util
{

template<typename Type>
__host__ __device__ Type min(Type a, Type b);

template<typename Type>
__host__ __device__ Type max(Type a, Type b);

}

}

