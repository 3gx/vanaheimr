/*	\file   HostReflection.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Saturday July 16, 2011
	\brief  The source file for the HostReflection set of functions.
*/

#pragma once

// Archaeopteryx Includes
#include <archaeopteryx/util/interface/HostReflection.h>
#include <archaeopteryx/util/interface/ThreadId.h>
#include <archaeopteryx/util/interface/StlFunctions.h>
#include <archaeopteryx/util/interface/debug.h>

// Standard Library Includes
#include <cstring>

namespace util
{

// TODO Remove these when __device__ can be embedded in a clas
__device__ HostReflection::Queue* _hostToDevice;
__device__ HostReflection::Queue* _deviceToHost;

__device__ void HostReflection::sendSynchronous(const Message& m)
{
	bool success = false;
	
	unsigned int bytes = m.payloadSize();

	char* buffer = new char[bytes + sizeof(SynchronousHeader)];
	
	SynchronousHeader* header = reinterpret_cast<SynchronousHeader*>(buffer);
	
	header->type     = Synchronous;
	header->threadId = threadId();
	header->handler  = m.handler();
	
	bool* flag = new bool;
	*flag = false;

	header->address = flag;	

	std::memcpy(buffer + sizeof(SynchronousHeader), m.payload(),
		m.payloadSize());
	
	while(!success)
	{
		success = _deviceToHost->push(buffer, bytes);
	}
	
	while(!flag);
	
	delete[] buffer;
}

__device__ void HostReflection::receive(Message& m)
{
	while(!_hostToDevice->peek());

	size_t bytes = m.payloadSize() + sizeof(Header);

	char* buffer = new char[bytes];
	
	_hostToDevice->pull(buffer, bytes);

	std::memcpy(m.payload(), buffer, m.payloadSize());

	delete[] buffer;
}

__host__ __device__ size_t HostReflection::maxMessageSize()
{
	return 64;
}

__device__ HostReflection::Queue::Queue(size_t size)
: _begin(new char[size]), _mutex((size_t)-1)
{
	_end  = _begin + size;
	_head = _begin;
	_tail = _begin;
}

__device__ HostReflection::Queue::~Queue()
{
	delete[] _begin;
}

__host__ __device__ bool HostReflection::Queue::push(
	const void* data, size_t size)
{
	if(size > _capacity()) return false;

	if(!_lock()) return false;	

	size_t remainder = _end - _head;
	size_t firstCopy = min(remainder, size);

	std::memcpy(_head, data, firstCopy);

	bool secondCopyNecessary = firstCopy != size;

	size_t secondCopy = secondCopyNecessary ? size - firstCopy : 0;
	
	std::memcpy(_begin, (char*)data + firstCopy, secondCopy);
	_head = secondCopyNecessary ? _begin + secondCopy : _head + firstCopy;

	_unlock();
	
	return true;
}

__host__ __device__ bool HostReflection::Queue::pull(
	void* data, size_t size)
{
	assert(size <= _used());

	if(!_lock()) return false;
	
	_tail = _read(data, size);

	_unlock();
	
	return true;
}

__host__ __device__ bool HostReflection::Queue::peek()
{
	if(!_lock()) return false;
	
	Header header;
	
	_read(&header, sizeof(Header));
	
	_unlock();
	
	return header.threadId == threadId();
}

__host__ __device__ size_t HostReflection::Queue::size() const
{
	return _end - _begin;
}

__host__ __device__  size_t HostReflection::Queue::_capacity() const
{
	size_t greaterOrEqual = _head - _tail;
	size_t less           = (_tail - _begin) + (_end - _head);
	
	bool isGreaterOrEqual = _head >= _tail;
	
	return (isGreaterOrEqual) ? greaterOrEqual : less;
}

__host__ __device__  size_t HostReflection::Queue::_used() const
{
	return size() - _capacity();
}

__host__ __device__ bool HostReflection::Queue::_lock()
{
	assert(_mutex != threadId());

	size_t result = atomicCAS(&_mutex, (size_t)-1, threadId());
	
	return result == threadId();
}

__host__ __device__ void HostReflection::Queue::_unlock()
{
	assert(_mutex == threadId());
	
	_mutex = (size_t)-1;
}

__host__ __device__ char* HostReflection::Queue::_read(
	void* data, size_t size)
{
	size_t remainder = _end - _tail;
	size_t firstCopy = min(remainder, size);

	std::memcpy(data, _tail, firstCopy);

	bool secondCopyNecessary = firstCopy != size;

	size_t secondCopy = secondCopyNecessary ? size - firstCopy : 0;
	
	std::memcpy((char*)data + firstCopy, _begin, secondCopy);
	
	return secondCopyNecessary ? _begin + secondCopy : _tail + firstCopy;
}

__global__ void _bootupHostReflection()
{
	_hostToDevice = new HostReflection::Queue(
		HostReflection::maxMessageSize() * 2);
	_deviceToHost = new HostReflection::Queue(
		HostReflection::maxMessageSize() * 2);
}

__host__ HostReflection::BootUp::BootUp()
{
	_bootupHostReflection<<<1, 1>>>();
	_kill   = false;
	_thread = new boost::thread(_run, &_kill);
}

__global__ void _teardownHostReflection()
{
	delete _hostToDevice;
	delete _deviceToHost;
}

__host__ HostReflection::BootUp::~BootUp()
{
	_kill = true;
	_thread->join();
	delete _thread;
	_teardownHostReflection<<<1, 1>>>();
}

__host__ void HostReflection::BootUp::_run(bool* kill)
{
	while(!*kill)
	{
		if(!handleMessage())
		{
			boost::thread::yield();
		}
	}
}

HostReflection::BootUp HostReflection::_booter;

}

