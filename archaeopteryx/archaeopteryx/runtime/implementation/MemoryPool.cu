/*	\file   MemoryPool.cu
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   November 15, 2012
	\brief  The source file for the MemoryPool class
*/

// Archaeopteryx Includes
#include <archaeopteryx/runtime/interface/MemoryPool.h>

namespace archaeopteryx
{

namespace rt
{

__device__ bool MemoryPool::allocate(uint64_t size, Address address)
{
	PageMap::iterator page = _pages.lower_bound(address);

	if(page != _pages.end())
	{
		if(page->second.endAddress() > address)
		{
			return false;
		}
	}

	_pages.insert(util::make_pair(address, Page(address, size)));

	return true;
}

__device__ MemoryPool::Address MemoryPool::allocate(uint64_t size)
{
	// Get the next available address
	Address address = 0;

	// TODO use a more efficient algorithm here
	for(PageMap::iterator page = _pages.begin(); page != _pages.end(); ++page)
	{
		if(address + size <= page->second.address())
		{
			break;
		}

		address = page->second.endAddress();
	}

	_pages.insert(util::make_pair(address, Page(address, size)));
	
	return address;
}

__device__ MemoryPool::Address MemoryPool::translate(Address address)
{
	PageMap::iterator page = _pages.lower_bound(address);

	if(page == _pages.end()) return InvalidAddress;

	if(address < page->second.endAddress())
	{
		return address - page->second.address() +
			page->second.physicalAddress();
	}
	
	return InvalidAddress;
}

__device__ MemoryPool::Page::Page(uint64_t size, Address address)
: _address(address), _data(size)
{

}

__device__ MemoryPool::Address MemoryPool::Page::address() const
{
	return _address;
}

__device__ MemoryPool::Address MemoryPool::Page::endAddress() const
{
	return address() + size();
}

__device__ MemoryPool::Address MemoryPool::Page::physicalAddress() const
{
	return (Address)_data.data();
}

__device__ uint64_t MemoryPool::Page::size() const
{
	return _data.size();
}

}

}

