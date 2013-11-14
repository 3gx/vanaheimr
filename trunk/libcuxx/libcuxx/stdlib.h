#pragma once

typedef struct {
  int quot;
  int rem;
} div_t;

typedef struct {
  long int quot;
  long int rem;
} ldiv_t;

struct lldiv_t {
  lldiv_t(long long int q, long long int r) : quot(q), rem(r) {}

  long long int quot;
  long long int rem;
} ;


extern long      a64l(const char *);
extern void      abort(void);
extern int       abs(int);
extern int       atexit(void (*)(void));
extern double    atof(const char *);
extern int       atoi(const char *);
extern long int  atol(const char *);
extern long long int  atoll(const char *);
extern void     *bsearch(const void *, const void *, size_t, size_t,
              int (*)(const void *, const void *));
extern void     *calloc(size_t, size_t);
extern div_t     div(int, int);
extern double    drand48(void);
extern char     *ecvt(double, int, int *, int *);
extern double    erand48(unsigned short int[3]);
extern void      exit(int);
extern char     *fcvt (double, int, int *, int *);
extern void      free(void *);


extern char     *gcvt(double, int, char *);
extern char     *getenv(const char *);
extern int       getsubopt(char **, char *const *, char **);
extern int       grantpt(int);
extern char     *initstate(unsigned int, char *, size_t);
extern long int  jrand48(unsigned short int[3]);
extern char     *l64a(long);
extern long int  labs(long int);
extern long long int llabs(long long int);
extern void      lcong48(unsigned short int[7]);
extern ldiv_t    ldiv(long int, long int);
extern lldiv_t    lldiv(long long int, long long int);
extern long int  lrand48(void);
extern void     *malloc(size_t);
extern int       mblen(const char *, size_t);
extern size_t    mbstowcs(wchar_t *, const char *, size_t);
extern int       mbtowc(wchar_t *, const char *, size_t);
extern char     *mktemp(char *);
extern int       mkstemp(char *);
extern long int  mrand48(void);
extern long int  nrand48(unsigned short int [3]);
extern char     *ptsname(int);
extern int       putenv(char *);
extern void      qsort(void *, size_t, size_t, int (*)(const void *,
              const void *));
extern int       rand(void);
extern int       rand_r(unsigned int *);
extern long      random(void);
extern void     *realloc(void *, size_t);
extern char     *realpath(const char *, char *);
extern unsigned  short int    seed48(unsigned short int[3]);
extern void      setkey(const char *);
extern char     *setstate(const char *);
extern void      srand(unsigned int);
extern void      srand48(long int);
extern void      srandom(unsigned);
extern double    strtod(const char *, char **);
extern long double    strtold(const char *, char **);
extern float     strtof(const char *, char **);
extern long int  strtol(const char *, char **, int);
extern long long int  strtoll(const char *, char **, int);

extern unsigned long int
          strtoul(const char *, char **, int);
extern unsigned long long int
          strtoull(const char *, char **, int);
extern int       system(const char *);
extern int       ttyslot(void);
extern int       unlockpt(int);
extern void     *valloc(size_t);
extern size_t    wcstombs(char *, const wchar_t *, size_t);
extern int       wctomb(char *, wchar_t);

extern void _Exit (int status);

