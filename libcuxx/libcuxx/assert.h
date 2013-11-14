#pragma once


# define __ASSERT_VOID_CAST static_cast<void>

#ifdef	NDEBUG

# define assert(expr) (__ASSERT_VOID_CAST (0))

#else /* Not NDEBUG.  */

/* This prints an "Assertion failed" message and aborts.  */
extern void __assert_fail (const char *__assertion, const char *__file,
			   unsigned int __line, const char *__function)
     __attribute__ ((__noreturn__));

/* Likewise, but prints the error text for ERRNUM.  */
extern void __assert_perror_fail (int __errnum, const char *__file,
				  unsigned int __line, const char *__function)
     __attribute__ ((__noreturn__));


/* The following is not at all used here but needed for standard
   compliance.  */
extern void __assert (const char *__assertion, const char *__file, int __line)
     __attribute__ ((__noreturn__));

# define assert(expr)							\
  ((expr)								\
   ? __ASSERT_VOID_CAST (0)						\
   : __assert_fail (#expr, __FILE__, __LINE__, __func__ ))

#endif /* NDEBUG.  */

