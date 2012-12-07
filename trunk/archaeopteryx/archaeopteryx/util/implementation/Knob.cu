/*	\file   Knob.cpp
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   November 7, 2012
	\brief  The source file for the Knob class
*/

// Archaeopteryx Includes
#include <archaeopteryx/util/interface/Knob.h>

#include <archaeopteryx/util/interface/map.h>
#include <archaeopteryx/util/interface/debug.h>

namespace archaeopteryx
{

namespace util
{

__device__ KnobBase::KnobBase(const util::string& name)
: _name(name)
{

}

__device__ KnobBase::~KnobBase()
{

}

__device__ const util::string& KnobBase::name() const
{
	return _name;
}

template<typename T>
__device__ Knob<T>::Knob(const util::string& name, const value_type& v)
: KnobBase(name), value(v)
{

}

template<typename T>
__device__ Knob<T>& Knob<T>::operator=(const value_type& v)
{
	value = v;
	
	return *this;
}


typedef util::map<util::string, KnobBase*> KnobMap;

static __device__ KnobMap* knobDatabaseImplementation = 0;

__device__ void KnobDatabase::addKnob(KnobBase* base)
{
	knobDatabaseImplementation->insert(util::make_pair(base->name(), base));
}

__device__ void KnobDatabase::removeKnob(const KnobBase& base)
{
	KnobMap::iterator knob = knobDatabaseImplementation->find(base.name());

	if(knob == knobDatabaseImplementation->end())
	{
		delete knob->second;
		knobDatabaseImplementation->erase(knob);
	}
}

__device__ const KnobBase& KnobDatabase::getKnobBase(const util::string& name)
{
	KnobMap::iterator knob = knobDatabaseImplementation->find(name);

	device_assert(knob != knobDatabaseImplementation->end());
	
	return *knob->second;
}

__device__ void KnobDatabase::loadDatabase()
{
	knobDatabaseImplementation = new KnobMap;
}

}

}


