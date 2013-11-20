/*! \file   CudaDriver.cpp
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   Wednesday November 13, 2013
	\brief  The source file for the CudaDriver class.
*/

// GPU Native Includes
#include <gpu-native/driver/interface/CudaDriver.h>

namespace gpunative
{

namespace driver
{

void CudaDriver::load()
{
	_interface.load();
}

bool CudaDriver::loaded()
{
	return _interface.loaded();
}

void CudaDriver::cuInit(unsigned int f)
{
	_check();
	
	_checkResult((*_interface.cuInit)(f));
}

void CudaDriver::cuDriverGetVersion(int* v)
{
	_check();
	
	_checkResult((*_interface.cuDriverGetVersion)(v));
}

void CudaDriver::cuDeviceGet(CUdevice* d, int o)
{
	_check();
	
	_checkResult((*_interface.cuDeviceGet)(d, o));
}

void CudaDriver::cuDeviceGetCount(int* c)
{
	_check();
	
	_checkResult((*_interface.cuDeviceGetCount)(c));
}

void CudaDriver::cuDeviceGetName(char* n, int l, CUdevice d)
{
	_check();
	
	_checkResult((*_interface.cuDeviceGetName)(n, l, d));
}

void CudaDriver::cuDeviceComputeCapability(int* m, int* minor, CUdevice d)
{
	_check();
	
	_checkResult((*_interface.cuDeviceComputeCapability)(m, minor, d));
}

void CudaDriver::cuDeviceTotalMem(size_t* b, CUdevice d)
{
	_check();
	
	_checkResult((*_interface.cuDeviceTotalMem)(b, d));
}

void CudaDriver::cuDeviceGetProperties(CUdevprop* prop, CUdevice dev)
{
	_check();
	
	_checkResult((*_interface.cuDeviceTotalMem)(b, d));
}

void CudaDriver::cuDeviceGetAttribute(int* p, CUdevice_attribute a, CUdevice d)
{
	_check();
	
	_checkResult((*_interface.cuDeviceGetAttribute)(p, a, d));
}

void CudaDriver::cuCtxCreate(CUcontext* c, unsigned int f, CUdevice d)
{
	_check();
	
	_checkResult((*_interface.cuCtxCreate)(c, f, d));
}

void CudaDriver::cuCtxGetApiVersion(CUcontext c, unsigned int* v)
{
	_check();
	
	_checkResult((*_interface.cuCtxGetApiVersion)(c, v));
}

void CudaDriver::cuCtxDestroy(CUcontext c)
{
	_check();
	
	_checkResult((*_interface.cuCtxDestroy)(c));
}

void CudaDriver::cuCtxSynchronize(void)
{
	_check();
	
	_checkResult((*_interface.cuCtxSynchronize)());
}

void CudaDriver::cuModuleLoadDataEx(CUmodule* m, 
		const void* i, unsigned int n, 
		CUjit_option* o, void** v)
{
	_check();
	
	_checkResult((*_interface.cuModuleLoadDataEx)(m, i, n, o, v));
}
void CudaDriver::cuModuleUnload(CUmodule h)
{
	_check();
	
	_checkResult((*_interface.cuModuleUnload)(h));
}
void CudaDriver::cuModuleGetFunction(CUfunction* f, CUmodule m, const char* n)
{
	_check();
	
	_checkResult((*_interface.cuModuleGetFunction)(f, m, n));
}
void CudaDriver::cuModuleGetGlobal(CUdeviceptr* p, 
		size_t* b, CUmodule m, const char* n)
{
	_check();
	
	_checkResult((*_interface.cuModuleGetGlobal)(p, b, m, n));
}

void CudaDriver::cuMemGetInfo(size_t* free, size_t* total)
{
	_check();
	
	_checkResult((*_interface.cuMemGetInfo)(free, total));
}

void CudaDriver::cuMemAlloc(CUdeviceptr* p, unsigned int b)
{
	_check();
	
	_checkResult((*_interface.cuMemAlloc)(p, b));
}

void CudaDriver::cuMemFree(CUdeviceptr p)
{
	_check();
	
	_checkResult((*_interface.cuMemAlloc)(p));
}

void CudaDriver::cuMemGetAddressRange(CUdeviceptr* p, size_t* d, CUdeviceptr dp)
{
	_check();
	
	_checkResult((*_interface.cuMemGetAddressRange)(p, d, dp));
}

void CudaDriver::cuMemAllocHost(void** p, unsigned int b)
{
	_check();
	
	_checkResult((*_interface.cuMemAllocHost)(p, b));
}

void CudaDriver::cuMemFreeHost(void* p)
{
	_check();
	
	_checkResult((*_interface.cuMemFreeHost)(p));
}

void CudaDriver::cuMemHostAlloc(void** pp, 
		unsigned long long bytesize, unsigned int Flags);
void CudaDriver::cuMemHostRegister(void* pp, 
		unsigned long long bytesize, unsigned int Flags);
void CudaDriver::cuMemHostUnregister(void* pp);

void CudaDriver::cuMemHostGetDevicePointer(CUdeviceptr* pdptr, 
		void* p, unsigned int Flags);
void CudaDriver::cuMemHostGetFlags(unsigned int* pFlags, void* p);
void CudaDriver::cuMemcpyHtoD (CUdeviceptr dstDevice, 
		const void* srcHost, unsigned int ByteCount);
void CudaDriver::cuMemcpyDtoH (void* dstHost, CUdeviceptr srcDevice, 
		unsigned int ByteCount);
void CudaDriver::cuFuncSetBlockShape (CUfunction hfunc, int x, 
		int y, int z);
void CudaDriver::cuFuncSetSharedSize (CUfunction hfunc, 
		unsigned int bytes);

void CudaDriver::cuParamSetSize(CUfunction hfunc, 
		unsigned int numbytes);
void CudaDriver::cuParamSetv(CUfunction hfunc, int offset, 
		void*  ptr, unsigned int numbytes);
void CudaDriver::cuLaunchGrid (CUfunction f, int grid_width, 
		int grid_height);
void CudaDriver::cuEventCreate(CUevent* phEvent, 
		unsigned int Flags);
void CudaDriver::cuEventRecord(CUevent hEvent, CUstream hStream);
void CudaDriver::cuEventQuery(CUevent hEvent);
void CudaDriver::cuEventSynchronize(CUevent hEvent);
void CudaDriver::cuEventDestroy(CUevent hEvent);
void CudaDriver::cuEventElapsedTime(float* pMilliseconds, 
		CUevent hStart, CUevent hEnd);
void CudaDriver::cuStreamCreate(CUstream* phStream, 
		unsigned int Flags);
void CudaDriver::cuStreamQuery(CUstream hStream);
void CudaDriver::cuStreamSynchronize(CUstream hStream);
void CudaDriver::cuStreamDestroy(CUstream hStream);
std::string CudaDriver::toString(CUresult result);

bool CudaDriver::doesFunctionExist(CUmodule hmod, const char* name);

void CudaDriver::_check()
{
	load();
	
	if(!loaded())
	{
		throw std::runtime_error("Tried to call libcuda function when "
			"the library is not loaded. Loading library failed, consider "
			"installing libcuda or putting it on your library search path.");
	}
}

static std::string errorCodeToString(CUresult r)
{
	std::stringstream stream;
	
	stream << r;
	
	// TODO: add more info
	
	return stream.str();
}

void CudaDriver::_checkResult(CUresult r)
{
	if(r != 0)
	{
		throw std::runtime_error("libcuda API call returned error code: " +
			errorCodeToString(r));
	}
}

CudaDriver::Interface::Interface()
: _library(nullptr)
{
	
}

CudaDriver::Interface::~Interface()
{
	unload();
}

static void checkFunction(void* pointer, const std::string& name)
{
	if(pointer == nullptr)
	{
		throw std::runtime_error("Failed to load function '" + name +
			"' from dynamic library.");
	}
}

void CudaDriver::Interface::load()
{
	if(loaded()) return;
	
    #ifdef __APPLE__
    const char* libraryName = "libcuda.dylib";
    #else
    const char* libraryName = "libcuda.so";
    #endif

	_library = dlopen(libraryName, RTLD_LAZY);

    util::log("CudaDriver") << "Loading library '" << libraryName << "'\n";

    if(!loaded())
	{
		return;
	}
	
	#define DynLink( function ) util::bit_cast(function, dlsym(_library, #function)); checkFunction((void*)function, #function)
	
	DynLink(cuInit);
	DynLink(cuDriverGetVersion);
	DynLink(cuDeviceGet);
	DynLink(cuDeviceGetCount);
	DynLink(cuDeviceGetName);
	DynLink(cuDeviceComputeCapability);
	DynLink(cuDeviceTotalMem);
	DynLink(cuDeviceGetProperties);
	DynLink(cuDeviceGetAttribute);
	DynLink(cuCtxCreate);
	DynLink(cuCtxDestroy);

	DynLink(cuModuleLoadDataEx);
	DynLink(cuModuleUnload);
	DynLink(cuModuleGetFunction);
	DynLink(cuModuleGetGlobal);
	DynLink(cuFuncSetBlockShape);
	DynLink(cuFuncSetSharedSize);

	DynLink(cuMemAlloc);
	DynLink(cuMemFree);
	DynLink(cuMemAllocHost);
	DynLink(cuMemFreeHost);
	DynLink(cuMemHostAlloc);
	DynLink(cuMemHostRegister);
	DynLink(cuMemHostUnregister);

	DynLink(cuMemHostGetDevicePointer);
	DynLink(cuMemcpyHtoD);
	DynLink(cuMemcpyDtoH);

	DynLink(cuParamSetv);
	DynLink(cuLaunchGrid);
	DynLink(cuEventCreate);
	DynLink(cuEventRecord);
	DynLink(cuEventQuery);
	DynLink(cuEventSynchronize);
	DynLink(cuEventDestroy);
	DynLink(cuEventElapsedTime);
	
	#undef DynLink	

	util::log("CudaDriver") << " success\n";
}

bool CudaDriver::Interface::loaded() const
{
	return _library != nullptr;
}

void CudaDriver::Interface::unload()
{
	if(!loaded()) return;

	dlclose(_library);
	_library = nullptr;
}

}

}


