/*! \file   CudaDriver.h
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Wednesday November 13, 2013
	\brief  The header file for the CudaDriver class.
*/

#pragma once

namespace gpunative
{

namespace driver
{

class CudaDriver
{
public:
	static void load();
	static bool loaded();

private:
	static void _check();

private:
	class Interface
	{
	public:
		CUresult (*cuInit)(unsigned int Flags);
		CUresult (*cuDriverGetVersion)(int *driverVersion);
		CUresult (*cuDeviceGet)(CUdevice *device, int ordinal);
		CUresult (*cuDeviceGetCount)(int *count);
		CUresult (*cuDeviceGetName)(char *name, int len, CUdevice dev);
		CUresult (*cuDeviceComputeCapability)(int *major,
			int *minor, CUdevice dev);
		CUresult (*cuDeviceTotalMem)(size_t *bytes, 
			CUdevice dev);
		CUresult (*cuDeviceGetProperties)(CUdevprop *prop, 
			CUdevice dev);
		CUresult (*cuDeviceGetAttribute)(int *pi, 
			CUdevice_attribute attrib, CUdevice dev);
		CUresult (*cuCtxCreate)(CUcontext *pctx, 
			unsigned int flags, CUdevice dev );
		CUresult (*cuCtxDestroy)( CUcontext ctx );

		CUresult (*cuModuleLoadDataEx)(CUmodule *module, 
			const void *image, unsigned int numOptions, 
			CUjit_option *options, void **optionValues);
		CUresult (*cuModuleUnload)(CUmodule hmod);
		CUresult (*cuModuleGetFunction)(CUfunction *hfunc, 
			CUmodule hmod, const char *name);
		CUresult (*cuModuleGetGlobal)(CUdeviceptr *dptr, 
			size_t *bytes, CUmodule hmod, const char *name);
		CUresult (*cuFuncSetBlockShape)(CUfunction hfunc, int x, 
			int y, int z);
		CUresult (*cuFuncSetSharedSize)(CUfunction hfunc, 
			unsigned int bytes);

		CUresult (*cuMemAlloc)( CUdeviceptr *dptr, 
			unsigned int bytesize);
		CUresult (*cuMemFree)(CUdeviceptr dptr);
		CUresult (*cuMemAllocHost)(void **pp, 
			unsigned int bytesize);
		CUresult (*cuMemFreeHost)(void *p);
		CUresult (*cuMemHostAlloc)(void **pp, 
			unsigned long long bytesize, unsigned int Flags );
		CUresult (*cuMemHostRegister)(void *pp, 
			unsigned long long bytesize, unsigned int Flags );
		CUresult (*cuMemHostUnregister)(void *pp);

		CUresult (*cuMemHostGetDevicePointer)( CUdeviceptr *pdptr, 
			void *p, unsigned int Flags );
		CUresult (*cuMemcpyHtoD)(CUdeviceptr dstDevice, 
			const void *srcHost, unsigned int ByteCount );
		CUresult (*cuMemcpyDtoH)(void *dstHost, 
			CUdeviceptr srcDevice, unsigned int ByteCount );
		
		CUresult (*cuParamSetv)(CUfunction hfunc, int offset, 
			void * ptr, unsigned int numbytes);
		CUresult (*cuLaunchGrid)(CUfunction f, int grid_width, 
			int grid_height);
		CUresult (*cuEventCreate)( CUevent *phEvent, 
			unsigned int Flags );
		CUresult (*cuEventRecord)( CUevent hEvent, 
			CUstream hStream );
		CUresult (*cuEventQuery)( CUevent hEvent );
		CUresult (*cuEventSynchronize)( CUevent hEvent );
		CUresult (*cuEventDestroy)( CUevent hEvent );
		CUresult (*cuEventElapsedTime)( float *pMilliseconds, 
			CUevent hStart, CUevent hEnd );
		
	public:
		/*! \brief The constructor zeros out all of the pointers */
		Interface();
		
		/*! \brief The destructor closes dlls */
		~Interface();
		/*! \brief Load the library */
		void load();
		/*! \brief Has the library been loaded? */
		bool loaded() const;
		/*! \brief unloads the library */
		void unload();
		
	private:
		void* _library;
		bool  _failed;

	};
	
private:
	static Interface _interface;

};

}

}


