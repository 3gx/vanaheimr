/*! \file   GpuNativeMain.h
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Thursday April 3, 2014
	\brief  The header file for the GpuNativeMain interface.
*/

#pragma once

extern void setEmbeddedPTX(const char* ptx);
extern const char* getEmbeddedPTX();
extern int gpuNativeMain(int argc, const char** argv);



