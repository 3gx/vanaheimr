/*!
	\file Casts.h
	\date Monday October 19, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for a set of non-standard casts
*/

#pragma once

#include <gpu-native/util/interface/debug.h>

namespace gpunative
{

namespace util
{

template< typename To, typename From >
class UnionCast
{
	public:
		union
		{
			To to;
			From from;
		};
};


template< typename To, typename From >
To bit_cast( const From & from )
{
	UnionCast< To, From > cast;
	cast.to = 0;
	cast.from = from;
	return cast.to;
}

template< typename To, typename From >
void bit_cast(To& to, const From& from)
{
	to = bit_cast<To, From>(from);
}

}

}


