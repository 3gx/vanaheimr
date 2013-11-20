/*! \file   CudaDriverTypes.h
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Wednesday November 13, 2013
	\brief  The header file for the CudaDriverTypes class.
*/

#pragma once

namespace gpunative
{

namespace driver
{

// Primitive types
typedef unsigned long long CUdeviceptr; // assumes 64-bit
typedef int CUdevice;
typedef int CUresult;
typedef int CUdevice_attribute;
typedef int CUjit_option;

typedef struct CUctx_st*     CUcontext;
typedef struct CUmod_st*     CUmodule;
typedef struct CUfunc_st*    CUfunction;
typedef struct CUevent_st*   CUevent;
typedef struct CUstream_st*  CUstream;
typedef struct CUdevprop_st* CUdevprop;

// Enums
const int CU_MEMHOSTREGISTER_DEVICEMAP = 0x02;


}

}



