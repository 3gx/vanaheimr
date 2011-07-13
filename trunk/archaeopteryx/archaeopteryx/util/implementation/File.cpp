/*! \file   File.cpp
	\date   Sunday June 26, 2011
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The source file for the File class.
*/

#pragma once

// Archaeoperyx Includes
#include <archaeopteryx/util/interface/HostReflection.h>
#include <archaeopteryx/util/interface/string.h>

namespace util
{

__device__ File::File(const char* fileName)
{
	OpenMessage open(fileName);
	
	HostReflection::sendSynchronous(open);
	
	OpenReply reply;
	
	HostReflection::receive(reply);
	
	_handle = reply.handle();
	_size   = reply.size();
	_put    = 0;
	_get    = 0;
}

__device__ File::~File()
{
	if(_handle != -1)
	{
		TeardownMessage teardown(_handle);
	
		HostReflection::sendSynchronous(teardown);
	}
}

__device__ void File::write(const void* data, size_t size)
{
	if(_put + size > size())
	{
		size = size() - _put;
	}
	
	WriteMessage message(data, size, _put, _handle);
	
	HostReflection::sendSynchronous(message);
	
	_put += size;
}

__device__ void File::read(void* data, size_t size)
{
	if(_get + size > size())
	{
		size = size() - _get;
	}
	
	ReadMessage message(data, size, _get, _handle);
	
	HostReflection::sendSynchronous(message);
	
	_get += size;
}

__device__ size_t File::size() const
{
	return _size;
}
	
__device__ size_t File::tellg() const
{
	return _get;
}
	
__device__ size_t File::tellp() const
{
	return _put;
}

__device__ void File::seekg(size_t g)
{
	if(g > size())
	{
		g = size();
	}
	
	_get = g;
}

__device__ void File::seekp(size_t p)
{
	if(p > size())
	{
		p = size();
	}
	
	_put = p;
}

__device__ File::OpenMessage::OpenMessage(const char* f)
{
	strlcpy(_filename, f, 256);
}

__device__ File::OpenMessage::~OpenMessage()
{

}

__device__ void* File::OpenMessage::payload()
{
	return _filename;
}

__device__ size_t File::OpenMessage::payloadSize() const
{
	return sizeof(_filename);
}

__device__ HostReflection::HandlerId File::OpenMessage::handler() const
{
	return HostReflection::OpenFileMessageHandler;
}

__device__ File::OpenReply::OpenReply()
{

}

__device__ File::OpenReply::~OpenReply()
{

}

__device__ File::Handle File::OpenReply::handle() const
{
	return _data.handle;
}

__device__ size_t File::OpenReply::size()   const
{
	return _data.size;
}

__device__ void* File::OpenReply::payload()
{
	return &_data;
}

__device__ size_t File::OpenReply::payloadSize() const
{
	return sizeof(Payload);
}

__device__ HostReflection::HandlerId File::OpenReply::handler() const
{
	return HostReflection::InvalidHandler;
}

__device__ File::TeardownMessage::TeardownMessage(Handle h)
: _handle(h)
{

}

__device__ File::TeardownMessage::~TeardownMessage()
{
	
}

__device__ void* File::TeardownMessage::payload()
{
	return &_handle;
}

__device__ size_t File::TeardownMessage::payloadSize() const
{
	return sizeof(Handle);
}

__device__ HostReflection::HandlerId File::TeardownMessage::handler() const
{
	return HostReflection::TeardownFileMessageHandler;
}

__device__ File::WriteMessage::WriteMessage(const void* data, size_t size,
	size_t pointer, Handle handle)
{
	_payload.data    = data;
	_payload.size    = size;
	_payload.pointer = pointer;
	_payload.handle  = handle;
}

__device__ File::WriteMessage::~WriteMessage()
{

}

__device__ void* File::WriteMessage::payload()
{
	return &_payload;
}

__device__ size_t File::WriteMessage::payloadSize() const
{
	return sizeof(Payload);
}

__device__ HostReflection::HandlerId File::WriteMessage::handler() const
{
	return HostReflection::FileWriteMessageHandler;
}

__device__ File::ReadMessage::ReadMessage(void* data,
	size_t size, size_t pointer, Handle handle)
{
	_payload.data    = data;
	_payload.size    = size;
	_payload.pointer = pointer;
	_payload.handle  = handle;
}
	
__device__ File::ReadMessage::~ReadMessage()
{

}

__device__ void* File::ReadMessage::payload()
{
	return &_payload;
}

__device__ size_t File::ReadMessage::payloadSize() const
{
	return sizeof(Payload);
}

__device__ HostReflection::HandlerId File::ReadMessage::handler() const
{
	return HostReflection::FileReadMessageHandler;
}

}

