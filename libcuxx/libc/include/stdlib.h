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

extern "C" long      a64l(const char *);
extern "C" void      abort(void);
extern "C" int       abs(int);
extern "C" int       atexit(void (*)(void));
extern "C" double    atof(const char *);
extern "C" int       atoi(const char *);
extern "C" long int  atol(const char *);
extern "C" long long int  atoll(const char *);
extern "C" void     *bsearch(const void *, const void *, size_t, size_t,
              int (*)(const void *, const void *));
extern "C" void     *calloc(size_t, size_t);
extern "C" div_t     div(int, int);
extern "C" double    drand48(void);
extern "C" char     *ecvt(double, int, int *, int *);
extern "C" double    erand48(unsigned short int[3]);
extern "C" void      exit(int);
extern "C" char     *fcvt (double, int, int *, int *);
extern "C" void      free(void *);


extern "C" char     *gcvt(double, int, char *);
extern "C" char     *getenv(const char *);
extern "C" int       getsubopt(char **, char *const *, char **);
extern "C" int       grantpt(int);
extern "C" char     *initstate(unsigned int, char *, size_t);
extern "C" long int  jrand48(unsigned short int[3]);
extern "C" char     *l64a(long);
extern "C" long int  labs(long int);
extern "C" long long int llabs(long long int);
extern "C" void      lcong48(unsigned short int[7]);
extern "C" ldiv_t    ldiv(long int, long int);
extern lldiv_t    lldiv(long long int, long long int);
extern "C" long int  lrand48(void);
extern "C" void     *malloc(size_t);
extern "C" int       mblen(const char *, size_t);
extern "C" size_t    mbstowcs(wchar_t *, const char *, size_t);
extern "C" int       mbtowc(wchar_t *, const char *, size_t);
extern "C" char     *mktemp(char *);
extern "C" int       mkstemp(char *);
extern "C" long int  mrand48(void);
extern "C" long int  nrand48(unsigned short int [3]);
extern "C" char     *ptsname(int);
extern "C" int       putenv(char *);
extern "C" void      qsort(void *, size_t, size_t, int (*)(const void *,
              const void *));
extern "C" int       rand(void);
extern "C" int       rand_r(unsigned int *);
extern "C" long      random(void);
extern "C" void     *realloc(void *, size_t);
extern "C" char     *realpath(const char *, char *);
extern "C" unsigned  short int    seed48(unsigned short int[3]);
extern "C" void      setkey(const char *);
extern "C" char     *setstate(const char *);
extern "C" void      srand(unsigned int);
extern "C" void      srand48(long int);
extern "C" void      srandom(unsigned);
extern "C" double    strtod(const char *, char **);
extern "C" long double    strtold(const char *, char **);
extern "C" float     strtof(const char *, char **);
extern "C" long int  strtol(const char *, char **, int);
extern "C" long long int  strtoll(const char *, char **, int);

extern "C" unsigned long int
          strtoul(const char *, char **, int);
extern "C" unsigned long long int
          strtoull(const char *, char **, int);
extern "C" int       system(const char *);
extern "C" int       ttyslot(void);
extern "C" int       unlockpt(int);
extern "C" void     *valloc(size_t);
extern "C" size_t    wcstombs(char *, const wchar_t *, size_t);
extern "C" int       wctomb(char *, wchar_t);

extern "C" void _Exit (int status);

