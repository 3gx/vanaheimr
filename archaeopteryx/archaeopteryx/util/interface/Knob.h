/*	\file   Knob.h
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   November 7, 2012
	\brief  The header file for the Knob class
*/

#pragma once

#include <archaeopteryx/util/interface/map.h>
#include <archaeopteryx/util/interface/string.h>

namespace archaeopteryx
{

namespace util
{

class KnobBase
{
public:
	string name;

};

template<typename T>
class Knob : public KnobBase
{
public:
	typedef T value_type;
	typedef value_type& reference;

public:
	__device__ Knob(const string& name, const reference);

public:
	__device__ Knob& operator=(const Knob&);
	__device__ Knob& operator=(const reference);

public:
	value_type value;

};

class KnobDatabase
{
public:
	__device__ static void addKnob(const KnobBase* base);
	__device__ static void removeKnob(const KnobBase* base);

	__device__ static const KnobBase& getKnob(const string& name);

public:
	static __device__ void loadDatabase();

};

}

}


