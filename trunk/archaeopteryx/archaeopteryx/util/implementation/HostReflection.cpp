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
#include <cassert>

namespace util
{

// TODO Remove these when __device__ can be embedded in a clas
__device__ HostReflection::Queue* _hostToDevice;
__device__ HostReflection::Queue* _deviceToHost;

char* _hostToDeviceMemory;
char* _deviceToHostMemory;

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

__device__ size_t HostReflection::maxMessageSize()
{
	return 64;
}

__device__ HostReflection::Queue::Queue(char* data, size_t size)
: _begin(data), _mutex((size_t)-1)
{
	_end  = _begin + size;
	_head = _begin;
	_tail = _begin;
}

__device__ HostReflection::Queue::~Queue()
{
	delete[] _begin;
}

__device__ bool HostReflection::Queue::push(
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

__device__ bool HostReflection::Queue::pull(
	void* data, size_t size)
{
	device_assert(size <= _used());

	if(!_lock()) return false;
	
	_tail = _read(data, size);

	_unlock();
	
	return true;
}

__device__ bool HostReflection::Queue::peek()
{
	if(!_lock()) return false;
	
	Header header;
	
	_read(&header, sizeof(Header));
	
	_unlock();
	
	return header.threadId == threadId();
}

__device__ size_t HostReflection::Queue::size() const
{
	return _end - _begin;
}

__device__  size_t HostReflection::Queue::_capacity() const
{
	size_t greaterOrEqual = _head - _tail;
	size_t less           = (_tail - _begin) + (_end - _head);
	
	bool isGreaterOrEqual = _head >= _tail;
	
	return (isGreaterOrEqual) ? greaterOrEqual : less;
}

__device__  size_t HostReflection::Queue::_used() const
{
	return size() - _capacity();
}

__device__ bool HostReflection::Queue::_lock()
{
	device_assert(_mutex != threadId());

	size_t result = atomicCAS(&_mutex, (size_t)-1, threadId());
	
	return result == threadId();
}

__device__ void HostReflection::Queue::_unlock()
{
	device_assert(_mutex == threadId());
	
	_mutex = (size_t)-1;
}

__device__ char* HostReflection::Queue::_read(
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

__global__ void _bootupHostReflection(char* hostToDeviceMemory,
	char* deviceToHostMemory)
{
	size_t size = HostReflection::maxMessageSize() * 2;

	_hostToDevice = new HostReflection::Queue(hostToDeviceMemory, size);
	_deviceToHost = new HostReflection::Queue(deviceToHostMemory, size);
}

__host__ HostReflection::BootUp::BootUp()
{
	size_t size = HostReflection::maxMessageSize() * 2;

	_hostToDeviceMemory = new char[size];
	_deviceToHostMemory = new char[size];

	cudaHostRegister(_hostToDeviceMemory, size, 0);
	cudaHostRegister(_deviceToHostMemory, size, 0);

	char* hostToDeviceMemoryPointer = 0;
	char* deviceToHostMemoryPointer = 0;
	
	cudaHostGetDevicePointer(&hostToDeviceMemoryPointer,
		_hostToDeviceMemory, 0);
	cudaHostGetDevicePointer(&deviceToHostMemoryPointer,
		_deviceToHostMemory, 0);

	_bootupHostReflection<<<1, 1>>>(hostToDeviceMemoryPointer,
		deviceToHostMemoryPointer);
	_kill   = false;
	_thread = new boost::thread(_runThread, this);
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

__host__ bool HostReflection::BootUp::_handleMessage()
{
	if(!_deviceToHost->hostAny())
	{
		return false;
	}
	
	HostReflection::Header* header = _deviceToHost->hostHeader();
	
	HandlerMap::iterator handler = _handlers.find(header->handler);
	assert(handler != _handlers.end());
	
	HostReflection::Message* message = _deviceToHost->hostMessage();
	
	handler->second(message);
	
	_deviceToHost->hostPop();
	
	return true;
}

__host__ void HostReflection::BootUp::_run()
{
	while(!_kill)
	{
		if(!_handleMessage())
		{
			boost::thread::yield();
		}
	}
}

__host__ void HostReflection::BootUp::_runThread(BootUp* booter)
{
	booter->_run();
}

HostReflection::BootUp HostReflection::_booter;

}

