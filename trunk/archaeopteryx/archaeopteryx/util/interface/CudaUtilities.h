/*! \file   CudaUtilities.h
	\date   Saturday Feburary 26, 2011
	\author Gregory Diamos and Sudnya Diamos
		<gregory.diamos@gatech.edu, mailsudnya@gmail.com>
	\brief  A set of common CUDA functions.
*/

#pragma once

/*! \brief Common utility functions */
namespace util
{

template<typename T>
__device__ T getParameter(void* parameter, unsigned int byte = 0)
{
	return *(T*)((char*)parameter + byte);
}

class SharedMemoryWrapper
{
public:
    __shared__ char _data[16384];
    
public:
    template <typename T>
    static T* at(unsigned int element)
    {
        return ((T*)_data)[byte];
    }
    
};

}

