/*	\file   HostReflection.cpp
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\date   Saturday July 16, 2011
	\brief  The source file for the HostReflection set of functions.
*/

// Archaeopteryx Includes
#include <archaeopteryx/util/interface/HostReflection.h>
#include <archaeopteryx/util/interface/ThreadId.h>
#include <archaeopteryx/util/interface/cstring.h>
#include <archaeopteryx/util/interface/StlFunctions.h>
#include <archaeopteryx/util/interface/debug.h>

// Standard Library Includes
#include <cstring>
#include <cassert>
#include <fstream>



// Forward Declarations

namespace ocelot
{
	void launch(const std::string& moduleName, const std::string& kernelName);
}

// Preprocessor Macros
#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 1

namespace archaeopteryx
{

namespace util
{

// TODO Remove these when __device__ can be embedded in a clas
__device__ HostReflection::DeviceQueue* _hostToDevice;
__device__ HostReflection::DeviceQueue* _deviceToHost;

template <typename T>
__device__ T HostReflection::Payload::get(unsigned int i)
{
	T temp = 0;
	
	std::memcpy(&temp, data.data + data.indexes[i], sizeof(T));

	return temp;
}

__device__ HostReflection::KernelLaunchMessage::KernelLaunchMessage(
	unsigned int ctas, unsigned int threads,
	const char* name, const Payload& payload)
: _stringLength(util::strlen(name) + 1), _data(new char[payloadSize()])
{
	char* data = _data;
	
	std::memcpy(data, &payload.data, sizeof(PayloadData));
	data += sizeof(PayloadData);

	std::memcpy(data, &ctas, sizeof(unsigned int));
	data += sizeof(unsigned int);

	std::memcpy(data, &threads, sizeof(unsigned int));
	data += sizeof(unsigned int);
	
	std::memcpy(data, &_stringLength, sizeof(unsigned int));
	data += sizeof(unsigned int);
	
	std::memcpy(data, name, _stringLength);
	data += _stringLength;
}

__device__ HostReflection::KernelLaunchMessage::~KernelLaunchMessage()
{
	delete[] _data;
}

__device__ void* HostReflection::KernelLaunchMessage::payload() const
{
	return _data;
}

__device__ size_t HostReflection::KernelLaunchMessage::payloadSize() const
{
	return sizeof(unsigned int) * 3 + sizeof(Payload) + _stringLength;
}

__device__ HostReflection::HandlerId
	HostReflection::KernelLaunchMessage::handler() const
{
	return KernelLaunchMessageHandler;
}

__device__ void HostReflection::sendAsynchronous(const Message& m)
{
	unsigned int bytes = m.payloadSize() + sizeof(Header);

	char* buffer = new char[bytes];
	
	Header* header = reinterpret_cast<Header*>(buffer);
	
	header->type     = Asynchronous;
	header->threadId = threadIdx.x;
	header->size     = bytes;
	header->handler  = m.handler();
	
	std::memcpy(buffer + sizeof(Header), m.payload(), m.payloadSize());
	 
	device_report(" sending asynchronous gpu->host message "
		"(%d type, %d id, %d size, %d handler)\n", Asynchronous,	
		header->threadId, bytes, m.handler());
	
	while(!_deviceToHost->push(buffer, bytes));

	delete[] buffer;
}

__device__ void HostReflection::sendSynchronous(const Message& m)
{
	unsigned int bytes = m.payloadSize() + sizeof(SynchronousHeader);

	char* buffer = new char[bytes];
	
	SynchronousHeader* header = reinterpret_cast<SynchronousHeader*>(buffer);
	
	header->type     = Synchronous;
	header->threadId = threadIdx.x;
	header->size     = bytes;
	header->handler  = m.handler();
	
	volatile bool* flag = new bool;
	*flag = false;

	header->address = (void*)flag;	

	std::memcpy(buffer + sizeof(SynchronousHeader), m.payload(),
		m.payloadSize());
	 
	device_report(" sending synchronous gpu->host message "
		"(%d type, %d id, %d size, %d handler, %x flag)\n", Synchronous,	
		header->threadId, bytes, m.handler(), header->address);
	
	while(!_deviceToHost->push(buffer, bytes));

	device_report("  waiting for ack...\n");
	
	while(*flag == false);

	device_report("   ...received ack\n");
	
	delete flag;
	delete[] buffer;
}

__device__ void HostReflection::receive(Message& m)
{
	while(!_hostToDevice->peek());

	device_report(" receiving cpu->gpu message.");

	size_t bytes = m.payloadSize() + sizeof(Header);

	char* buffer = new char[bytes];
	
	_hostToDevice->pull(buffer, bytes);

	device_report("  bytes: %d\n", (int)(bytes - sizeof(Header)));

	std::memcpy(m.payload(), (buffer + sizeof(Header)), m.payloadSize());

	delete[] buffer;
}

__device__ void HostReflection::launch(unsigned int ctas, unsigned int threads,
	const char* functionName, const Payload& payload)
{
	KernelLaunchMessage message(ctas, threads, functionName, payload);

	sendAsynchronous(message);
}

__device__ unsigned int align(unsigned int address, unsigned int alignment)
{
	unsigned int remainder = address % alignment;
	return remainder == 0 ? address : address + (alignment - remainder);
}

template<typename T0, typename T1, typename T2, typename T3, typename T4>
__device__ HostReflection::Payload HostReflection::createPayload(const T0& t0,
	const T1& t1, const T2& t2, const T3& t3, const T4& t4)
{
	Payload result;

	PayloadData& payload = result.data;

	unsigned int index = 0;
	
	payload.indexes[0] = index;
	std::memcpy(payload.data + index, &t0, sizeof(T0));
	index += sizeof(T0);
	index =  align(index, sizeof(T1));
	
	payload.indexes[1] = index;
	std::memcpy(payload.data + index, &t1, sizeof(T1));
	index += sizeof(T1);
	index =  align(index, sizeof(T2));
	
	payload.indexes[2] = index;
	std::memcpy(payload.data + index, &t2, sizeof(T2));
	index += sizeof(T2);
	index =  align(index, sizeof(T3));
	
	payload.indexes[3] = index;
	std::memcpy(payload.data + index, &t3, sizeof(T3));
	index += sizeof(T3);
	index =  align(index, sizeof(T4));
	
	payload.indexes[4] = index;
	std::memcpy(payload.data + index, &t4, sizeof(T4));

	return result;
}

template<typename T0, typename T1, typename T2, typename T3>
__device__ HostReflection::Payload HostReflection::createPayload(const T0& t0,
	const T1& t1, const T2& t2, const T3& t3)
{
	return createPayload(t0, t1, t2, t3, (int)0);
}

template<typename T0, typename T1, typename T2>
__device__ HostReflection::Payload HostReflection::createPayload(const T0& t0,
	const T1& t1, const T2& t2)
{
	return createPayload(t0, t1, t2, (int)0);
}

template<typename T0, typename T1>
__device__ HostReflection::Payload HostReflection::createPayload(const T0& t0,
	const T1& t1)
{
	return createPayload(t0, t1, (int)0);
}

template<typename T0>
__device__ HostReflection::Payload HostReflection::createPayload(const T0& t0)
{
	return createPayload(t0, (int)0);
}

__device__ HostReflection::Payload HostReflection::createPayload()
{
	return createPayload((int)0);
}

__device__ size_t HostReflection::maxMessageSize()
{
	return 512;
}

__device__ HostReflection::DeviceQueue::DeviceQueue(QueueMetaData* m)
: _metadata(m)
{
	device_report("binding device queue to metadata (%d size, "
		"%d head, %d tail, %d mutex)\n", (int)m->size, (int)m->head,
		(int)m->tail, m->mutex);
}

__device__ HostReflection::DeviceQueue::~DeviceQueue()
{

}

__device__ bool HostReflection::DeviceQueue::push(const void* data, size_t size)
{
	device_assert(size <= this->size());

	if(size > _capacity()) return false;
	
	if(!_lock()) return false;	

	device_report("pushing %d bytes into gpu->cpu queue.\n", (int)size);

	size_t end  = _metadata->size;
	size_t head = _metadata->head;

	size_t remainder = end - head;
	size_t firstCopy = min(remainder, size);

	std::memcpy(_metadata->deviceBegin + head, data, firstCopy);

	bool secondCopyNecessary = firstCopy != size;

	size_t secondCopy = secondCopyNecessary ? size - firstCopy : 0;
	
	std::memcpy(_metadata->deviceBegin, (char*)data + firstCopy, secondCopy);
	_metadata->head = secondCopyNecessary ? secondCopy : head + firstCopy;
	
	device_report(" after push (%d used, %d remaining, %d size)\n",
		(int)_used(), (int)_capacity(), (int)this->size());
	
	_unlock();
	
	return true;
}

__device__ bool HostReflection::DeviceQueue::pull(void* data, size_t size)
{
	device_assert(size <= _used());

	if(!_lock()) return false;
	
	_metadata->tail = _read(data, size);

	_unlock();
	
	return true;
}

__device__ bool HostReflection::DeviceQueue::peek()
{
	if(_used() < sizeof(Header)) return false;

	if(!_lock()) return false;
	
	Header header;
	
	_read(&header, sizeof(Header));
	
	_unlock();
	
	return header.threadId == threadId();
}

__device__ size_t HostReflection::DeviceQueue::size() const
{
	return _metadata->size;
}

__device__  size_t HostReflection::DeviceQueue::_used() const
{
	size_t end  = _metadata->size;
	size_t head = _metadata->head;
	size_t tail = _metadata->tail;
	
	size_t greaterOrEqual = head - tail;
	size_t less           = (head) + (end - tail);
	
	bool isGreaterOrEqual = head >= tail;
	
	return (isGreaterOrEqual) ? greaterOrEqual : less;
}

__device__  size_t HostReflection::DeviceQueue::_capacity() const
{
	return size() - _used();
}

__device__ bool HostReflection::DeviceQueue::_lock()
{
	device_assert(_metadata->mutex != threadId());
	
	size_t result = atomicCAS((long long unsigned int*)&_metadata->mutex,
		(long long unsigned int)-1, (long long unsigned int)threadId());
	
	return result == (size_t)-1;
}

__device__ void HostReflection::DeviceQueue::_unlock()
{
	device_assert(_metadata->mutex == threadId());
	
	_metadata->mutex = (size_t)-1;
}

__device__ size_t HostReflection::DeviceQueue::_read(
	void* data, size_t size)
{
	size_t end  = _metadata->size;
	size_t tail = _metadata->tail;

	size_t remainder = end - tail;
	size_t firstCopy = min(remainder, size);

	std::memcpy(data, _metadata->deviceBegin + tail, firstCopy);

	bool secondCopyNecessary = firstCopy != size;

	size_t secondCopy = secondCopyNecessary ? size - firstCopy : 0;
	
	std::memcpy((char*)data + firstCopy, _metadata->deviceBegin, secondCopy);
	
	return secondCopyNecessary ? secondCopy : tail + firstCopy;
}

__global__ void _bootupHostReflection(
	HostReflection::QueueMetaData* hostToDeviceMetadata,
	HostReflection::QueueMetaData* deviceToHostMetadata)
{
	_hostToDevice = new HostReflection::DeviceQueue(hostToDeviceMetadata);
	_deviceToHost = new HostReflection::DeviceQueue(deviceToHostMetadata);
}

__global__ void _teardownHostReflection()
{
	delete _hostToDevice;
	delete _deviceToHost;
}

}

}

