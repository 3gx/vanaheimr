
#include <fcntl.h>
#include <assert.h>


extern int open(const char *, int, ...)
{
	assert(false && "not implemented");
	
	return 0;
}

extern int create(const char *, mode_t)
{
	assert(false && "not implemented");
	
	return 0;
}

extern int fcntl(int, int, ...)
{
	assert(false && "not implemented");
	
	return 0;
}

extern int flock(int, int)
{
	assert(false && "not implemented");
	
	return 0;
}


