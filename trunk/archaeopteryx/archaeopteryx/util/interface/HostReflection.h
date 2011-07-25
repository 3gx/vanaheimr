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
	typedef unsigned int HandlerId;

	class Message
	{
	public:
		__device__ virtual void* payload() const = 0;
		__device__ virtual size_t payloadSize() const = 0;
		__device__ virtual HostReflection::HandlerId handler() const = 0;
	};
	
	enum MessageHandler
	{
		OpenFileMessageHandler,
		TeardownFileMessageHandler,
		FileWriteMessageHandler,
		FileReadMessageHandler,
		InvalidMessageHandler = -1
	};
		
public:
	__device__ static void sendSynchronous(const Message& m);
	__device__ static void receive(Message& m);

public:
	class Queue
	{
	public:
		__device__ Queue(size_t size);
		__device__ ~Queue();
	
	public:
		__device__ bool push(const void* data, size_t size);
		__device__ bool pull(void* data, size_t size);

	public:
		__device__ bool peek();
		__device__ size_t size() const;
	
	private:
		char*  _begin;
		char*  _end;
		char*  _head;
		char*  _tail;
		size_t _mutex;
		
	private:
		__device__ size_t _capacity() const;
		__device__ size_t _used() const;
		
	private:
		__device__ bool _lock();
		__device__ void _unlock();
		__device__ char* _read(void* data, size_t size);

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
	class BootUp
	{
	public:
		__host__ BootUp();
		__host__ ~BootUp();
	};

private:
	static BootUp _booter;

};

}

// TODO remove this when we get a real linker
#include <archaeopteryx/util/implementation/HostReflection.cpp>

