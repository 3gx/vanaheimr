
// Standard Library Includes
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

void __assert_fail (const char *__assertion, const char *__file,
			   unsigned int __line, const char *__function)
{
	printf("%s:%s:%d assertion '%s' failed.\n", __file, __function, __line, __assertion);

	abort();
}
    
void __assert_perror_fail (int __errnum, const char *__file,
				  unsigned int __line, const char *__function)
{
	printf("%s:%s:%d assertion failed with error number %d.\n", __file, __function, __line, __errnum);

	abort();
}


void __assert (const char *__assertion, const char *__file, int __line)
{
	printf("%s:%d assertion '%s' failed.\n", __file, __line, __assertion);

	abort();
}


