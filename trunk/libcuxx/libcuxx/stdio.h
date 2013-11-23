
#pragma once

// Includes
#include <__parallel_config>
#include <stdarg.h>

// Macros
#define EOF -1

// Types
struct FILE;

typedef size_t fpos_t;
typedef size_t off_t;

// Functions
extern "C" int remove ( const char * filename );
extern "C" void clearerr ( FILE * stream );
extern "C" int fclose ( FILE * stream );
extern "C" int feof ( FILE * stream );
extern "C" int ferror ( FILE * stream );
extern "C" int fflush ( FILE * stream );
extern "C" int fgetc ( FILE * stream );
extern "C" int fgetpos ( FILE * stream, fpos_t * pos );
extern "C" char * fgets ( char * str, int num, FILE * stream );
extern "C" FILE * fopen ( const char * filename, const char * mode );
extern "C" int fprintf ( FILE * stream, const char * format, ... );
extern "C" int fputc ( int character, FILE * stream );
extern "C" int fputs ( const char * str, FILE * stream );
extern "C" size_t   fread(void *, size_t, size_t, FILE *);
extern "C" FILE    *freopen(const char *, const char *, FILE *);
extern "C" int      fscanf(FILE *, const char *, ...);
extern "C" int      fseek(FILE *, long int, int);
extern "C" int      fseeko(FILE *, off_t, int);
extern "C" int      fsetpos(FILE *, const fpos_t *);
extern "C" long int ftell(FILE *);
extern "C" off_t    ftello(FILE *);
extern "C" int      ftrylockfile(FILE *);
extern "C" void     funlockfile(FILE *);
extern "C" size_t   fwrite(const void *, size_t, size_t, FILE *);
extern "C" int      getc(FILE *);
extern "C" int      getchar(void);
extern "C" int      getc_unlocked(FILE *);
extern "C" int      getchar_unlocked(void);
extern "C" int      getopt(int, char * const[], const char);
extern "C" char    *gets(char *);
extern "C" int      getw(FILE *);
extern "C" int      pclose(FILE *);
extern "C" void     perror(const char *);
extern "C" FILE    *popen(const char *, const char *);
extern "C" int      putc(int, FILE *);
extern "C" int      putchar(int);
extern "C" int      putc_unlocked(int, FILE *);
extern "C" int      putchar_unlocked(int);
extern "C" int      puts(const char *);
extern "C" int      putw(int, FILE *);
extern "C" int      remove(const char *);
extern "C" int      rename(const char *, const char *);
extern "C" void     rewind(FILE *);
extern "C" int      scanf(const char *, ...);
extern "C" void     setbuf(FILE *, char *);
extern "C" int      setvbuf(FILE *, char *, int, size_t);
extern "C" int      snprintf(char *, size_t, const char *, ...);
extern "C" int      sprintf(char *, const char *, ...);
extern "C" int      sscanf(const char *, const char *, ...);
extern "C" char    *tempnam(const char *, const char *);
extern "C" FILE    *tmpfile(void);
extern "C" char    *tmpnam(char *);
extern "C" int      ungetc(int, FILE *);
extern "C" int vfscanf ( FILE * stream, const char * format, va_list arg );
extern "C" int vscanf ( const char * format, va_list arg );
extern "C" int      vfprintf(FILE *, const char *, va_list);
extern "C" int      vsnprintf(char *, size_t, const char *, va_list);
extern "C" int      vsprintf(char *, const char *, va_list);
extern "C" int vsscanf ( const char * s, const char * format, va_list arg );

// HACK use variadic templates to implement variadic functions
#if 1

inline int getSize()
{
	return 0;
}

template<typename Arg>
inline int getSize(Arg arg)
{
	return sizeof(arg);
}

template<typename Arg, typename... Args>
inline int getSize(Arg arg, Args... args)
{
	return sizeof(arg) + getSize(args...);
}

inline void fillBuffer(char* buffer, int offset)
{
}

template<typename Arg>
inline void fillBuffer(char* buffer, int offset, Arg arg)
{
	*reinterpret_cast<Arg*>(buffer + offset) = arg;
}

template<typename Arg, typename... Args>
inline void fillBuffer(char* buffer, int offset, Arg arg, Args... args)
{
	*reinterpret_cast<Arg*>(buffer + offset) = arg;

	fillBuffer(buffer, offset + sizeof(arg), args...);
}

extern "C" int vprintf(const char *, void*);

template<typename... Args>
int printf(const char* format, Args... args)
{
	const int size = getSize(args...);
	const int bufferSize = size > 32 ? size : 32;

	char buffer[bufferSize];
	fillBuffer(buffer, 0, args...);

	int done;

	//va_start(arg, format);
	done = vprintf (format, buffer);
	//va_end(arg);

	return done;
}
#else

extern "C" int      vprintf(const char *, va_list);
extern "C" int      printf(const char *, ...);
#endif


