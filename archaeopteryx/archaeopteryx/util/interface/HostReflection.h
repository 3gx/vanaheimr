/*	\file   HostReflection.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Saturday July 16, 2011
	\brief  The header file for the HostReflection set of functions.
*/

#pragma once

namespace util
{

class HostReflection
{
public:
	class Message
	{
	public:
		__device__ virtual void* payload() = 0
		__device__ virtual size_t payloadSize() const = 0
		__device__ virtual HostReflection::HandlerId handler() const = 0;
	};
	
	typedef unsigned int HandlerId;
		
public:
	__device__ static void sendSynchronous(const Message& m);
	__device__ static void receive(Message& m);

private:
	class Queue
	{
	public:
		__device__ Queue(size_t size);
		__device__ ~Queue(size_t size);
	
	public:
		__host__ __device__ bool push(const void* data, size_t size);
		__host__ __device__ bool pull(void* data, size_t size);

	public:
		__host__ __device__ bool peek();
	
	private:
		char*  _begin;
		char*  _end;
		char*  _head;
		char*  _tail;
		size_t _mutex;
		
	private:
		__host__ __device__ size_t _capacity() const;
		__host__ __device__ size_t _used() const;
		
	private:
		__host__ __device__ bool _lock();
		__host__ __device__ void _unlock();
		__host__ __device__ char* _read(void* data, size_t size);


	};

	enum MessageType
	{
		Synchronous,
		Asynchronous,
		Invalid,
	};

	class Header
	{
	public:
		MessageType  type;
		unsigned int threadId;
		HandlerId    handler;
	};
	
	class SynchronousHeader : public Header
	{
	public:
		void* address;
	};

private:
	static Queue _hostToDevice;
	static Queue _deviceToHost;

};

}

// TODO remove this when we get a real linker
#include <archaeopteryx/util/implementation/HostReflection.cpp>

