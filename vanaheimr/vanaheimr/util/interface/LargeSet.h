/*! \file   LargeSet.h
	\date   Friday September 14, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for the LargeSet class.
*/

#pragma once

namespace vanaheimr
{

namespace util
{

template<typename T>
class LargeSet : public std::set<T>
{

};

}

}


