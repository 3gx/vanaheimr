
#include <nl_types.h>
#include <assert.h>

extern int catclose(nl_catd)
{
	assert(false && "not implemented");	
	
	return 0;
}

extern char* catgets(nl_catd, int, int, const char *)
{
	assert(false && "not implemented");	
	
	return 0;
}

extern nl_catd catopen(const char *, int)
{
	assert(false && "not implemented");	
	
	return 0;
}

