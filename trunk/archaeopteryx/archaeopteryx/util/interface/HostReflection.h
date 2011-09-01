/*	\file   HostReflection.h
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Saturday July 16, 2011
	\brief  The header file for the HostReflection set of functions.
*/

#pragma once

// Boost Includes
#include <boost/thread.hpp>

// Standard Library Includes
#include <map>

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
	__host__ __device__ static size_t maxMessageSize();

public:
	__host__ static void create();
	__host__ static void destroy();

public:
	/*! \brief Handle an open message on the host */
	__host__ static void handleOpenFile(const Message*);
	
	/*! \brief Handle a teardown message on the host */
	__host__ static void handleTeardownFile(const Message*);
	
	/*! \brief Handle a file write message on the host */
	__host__ static void handleFileWrite(const Message*);
	
	/*! \brief Handle a file read message on the host */
	__host__ static void handleFileRead(const Message*);

public:
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
		unsigned int size;
		HandlerId    handler;
	};
	
	class SynchronousHeader : public Header
	{
	public:
		void* address;
	};

	class QueueMetaData
	{
	public:
		char*  hostBegin;
		char*  deviceBegin;

		size_t size;
		size_t head;
		size_t tail;
		size_t mutex;
	};

	class HostQueue
	{
	public:
		__host__ HostQueue(QueueMetaData* metadata);
		__host__ ~HostQueue();

	public:	
		__host__ Message* message();
		__host__ Header*  header();

	public:
		__host__ bool push(const void* data, size_t size);
		__host__ bool pull(void* data, size_t size);

	public:
		__host__ bool peek();
		__host__ size_t size() const;

	private:
		QueueMetaData* _metadata;

	private:
		__host__ size_t _capacity() const;
		__host__ size_t _used() const;

	private:
		__host__ size_t _read(void* data, size_t size);
	};

	class DeviceQueue
	{
	public:
		__device__ DeviceQueue(QueueMetaData* metadata);
		__device__ ~DeviceQueue();

	public:
		__device__ bool push(const void* data, size_t size);
		__device__ bool pull(void* data, size_t size);

	public:
		__device__ bool peek();
		__device__ size_t size() const;
	
	private:
		QueueMetaData* _metadata;
		
	private:
		__device__ size_t _capacity() const;
		__device__ size_t _used() const;
		
	private:
		__device__ bool _lock();
		__device__ void _unlock();
		__device__ size_t _read(void* data, size_t size);
	};

private:
	class BootUp
	{
	public:
		typedef void (*MessageHandler)(const Message*);
		typedef std::map<int, MessageHandler> HandlerMap;
	
	public:
		__host__ BootUp();
		__host__ ~BootUp();

	public:
		__host__ void addHandler(int handlerId, MessageHandler handler);
		
	private:
		boost::thread* _thread;
		HostQueue*     _hostToDeviceQueue;
		HostQueue*     _deviceToHostQueue;
		bool           _kill;
	
	private:
		HandlerMap _handlers;

	private:
		__host__ void _run();
		__host__ bool _handleMessage();
		__host__ void _addMessageHandlers();
	
	private:
		static void _runThread(BootUp* kill);
	};

private:
	static BootUp* _booter;

};

}

// TODO remove this when we get a real linker
#include <archaeopteryx/util/implementation/HostReflection.cpp>

