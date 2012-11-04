#pragma once

#ifdef __NVCC__
	typedef           unsigned char   uint8_t;
	typedef           unsigned short uint16_t;
	typedef           unsigned int   uint32_t;
	typedef long long unsigned int   uint64_t;
#else
	#include <cstdint>
#endif

