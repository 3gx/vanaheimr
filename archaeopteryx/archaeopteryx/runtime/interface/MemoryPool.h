/*	\file   MemoryPool.h
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   November 15, 2012
	\brief  The header file for the MemoryPool class
*/

#pragma once

// Archaeopteryx Includes
#include <archaeopteryx/util/interface/IntTypes.h>
#include <archaeopteryx/util/interface/map.h>
#include <archaeopteryx/util/interface/vector.h>

namespace archaeopteryx
{

namespace runtime
{

class MemoryPool
{
public:
	typedef uint64_t Address;

public:
	bool allocate(uint64_t size, Address address);
	void allocate(Address address);

	Address translate(Address address);

private:
	class Page
	{
	public:
		Page(uint64_t size, Address address);

	public:
		Address          address() const;
		Address       endAddress() const;
		Address  physicalAddress() const;
		uint64_t            size() const;

	private:
		typedef util::vector<uint8_t> DataVector;

	private:
		Address    _address;
		DataVector _data;	
	};


private:
	typedef util::map<Address, Page> PageMap;

private:
	PageMap _pages;	

};

}

}

