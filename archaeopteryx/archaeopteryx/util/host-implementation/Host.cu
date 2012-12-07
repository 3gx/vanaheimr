

#include <archaeopteryx/util/interface/Host.h>
#include <archaeopteryx/util/interface/HostReflection.h>

namespace archaeopteryx
{

namespace util
{

void startupHostReflection()
{
	HostReflection::create("module");
}

void teardownHostReflection()
{
	HostReflection::destroy();
}

}

}



