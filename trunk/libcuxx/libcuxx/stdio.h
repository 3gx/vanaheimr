
#pragma once

// Includes
#include <stdarg.h>

// Macros
#define EOF -1

// Types
struct FILE;

typedef size_t fpos_t;
typedef size_t off_t;

// Functions
extern int remove ( const char * filename );
extern void clearerr ( FILE * stream );
extern int fclose ( FILE * stream );
extern int feof ( FILE * stream );
extern int ferror ( FILE * stream );
extern int fflush ( FILE * stream );
extern int fgetc ( FILE * stream );
extern int fgetpos ( FILE * stream, fpos_t * pos );
extern char * fgets ( char * str, int num, FILE * stream );
extern FILE * fopen ( const char * filename, const char * mode );
extern int fprintf ( FILE * stream, const char * format, ... );
extern int fputc ( int character, FILE * stream );
extern int fputs ( const char * str, FILE * stream );
extern size_t   fread(void *, size_t, size_t, FILE *);
extern FILE    *freopen(const char *, const char *, FILE *);
extern int      fscanf(FILE *, const char *, ...);
extern int      fseek(FILE *, long int, int);
extern int      fseeko(FILE *, off_t, int);
extern int      fsetpos(FILE *, const fpos_t *);
extern long int ftell(FILE *);
extern off_t    ftello(FILE *);
extern int      ftrylockfile(FILE *);
extern void     funlockfile(FILE *);
extern size_t   fwrite(const void *, size_t, size_t, FILE *);
extern int      getc(FILE *);
extern int      getchar(void);
extern int      getc_unlocked(FILE *);
extern int      getchar_unlocked(void);
extern int      getopt(int, char * const[], const char);
extern char    *gets(char *);
extern int      getw(FILE *);
extern int      pclose(FILE *);
extern void     perror(const char *);
extern FILE    *popen(const char *, const char *);
extern int      printf(const char *, ...);
extern int      putc(int, FILE *);
extern int      putchar(int);
extern int      putc_unlocked(int, FILE *);
extern int      putchar_unlocked(int);
extern int      puts(const char *);
extern int      putw(int, FILE *);
extern int      remove(const char *);
extern int      rename(const char *, const char *);
extern void     rewind(FILE *);
extern int      scanf(const char *, ...);
extern void     setbuf(FILE *, char *);
extern int      setvbuf(FILE *, char *, int, size_t);
extern int      snprintf(char *, size_t, const char *, ...);
extern int      sprintf(char *, const char *, ...);
extern int      sscanf(const char *, const char *, ...);
extern char    *tempnam(const char *, const char *);
extern FILE    *tmpfile(void);
extern char    *tmpnam(char *);
extern int      ungetc(int, FILE *);
extern int vfscanf ( FILE * stream, const char * format, va_list arg );
extern int vscanf ( const char * format, va_list arg );
extern int      vfprintf(FILE *, const char *, va_list);
extern int      vprintf(const char *, va_list);
extern int      vsnprintf(char *, size_t, const char *, va_list);
extern int      vsprintf(char *, const char *, va_list);
extern int vsscanf ( const char * s, const char * format, va_list arg );

