/*	\file   Knob.h
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   November 7, 2012
	\brief  The header file for the Knob class
*/

#pragma once

namespace archaeopteryx
{

namespace util
{

class KnobBase

template<typename T>
class Knob : public KnobBase
{
public:
	typedef T value_type;
	typedef value_type& reference;

public:
	__device__ Knob(const char* name, const reference);

public:
	__device__ Knob& operator=(const Knob&);
	__device__ Knob& operator=(const reference);

public:
	value_type value;

};

class KnobDatabase
{
public:

	static __device__ addKnob(const KnobBase* base);
	static __device__ removeKnob(const KnobBase* base);
	
};

}

}


