#pragma once

/******************************************************************************
 * LIBCUXX Configuration                                                      *
 ******************************************************************************/

#ifndef __FLT_EVAL_METHOD
#define __FLT_EVAL_METHOD 0
#endif

/******************************************************************************
 * Floating point data types                                                  *
 ******************************************************************************/

/*  Define float_t and double_t per C standard, ISO/IEC 9899:2011 7.12 2,
    taking advantage of GCC's __FLT_EVAL_METHOD__ (which a compiler may
    define anytime and GCC does) that shadows FLT_EVAL_METHOD (which a
    compiler must define only in float.h).                                    */
#if __FLT_EVAL_METHOD__ == 0
    typedef float float_t;
    typedef double double_t;
#elif __FLT_EVAL_METHOD__ == 1
    typedef double float_t;
    typedef double double_t;
#elif __FLT_EVAL_METHOD__ == 2 || __FLT_EVAL_METHOD__ == -1
    typedef long double float_t;
    typedef long double double_t;
#else /* __FLT_EVAL_METHOD__ */
#   error "Unsupported value of __FLT_EVAL_METHOD__."
#endif /* __FLT_EVAL_METHOD__ */

#   define    HUGE_VAL     __builtin_huge_val()
#   define    HUGE_VALF    __builtin_huge_valf()
#   define    HUGE_VALL    __builtin_huge_vall()
#   define    NAN          __builtin_nanf("0x7fc00000")

#define INFINITY    HUGE_VALF

/******************************************************************************
 *      Taxonomy of floating point data types                                 *
 ******************************************************************************/

#define FP_NAN          1
#define FP_INFINITE     2
#define FP_ZERO         3
#define FP_NORMAL       4
#define FP_SUBNORMAL    5
#define FP_SUPERNORMAL  6 /* legacy PowerPC support; this is otherwise unused */

#if defined __ARM_VFPV4__
/*  On these architectures, fma(), fmaf( ), and fmal( ) are generally about as
    fast as (or faster than) separate multiply and add of the same operands.  */
#   define FP_FAST_FMA     1
#   define FP_FAST_FMAF    1
#   define FP_FAST_FMAL    1
#elif (defined __i386__ || defined __x86_64__) && defined __FMA__
/*  When targeting the FMA ISA extension, fma() and fmaf( ) are generally
    about as fast as (or faster than) separate multiply and add of the same
    operands, but fmal( ) may be more costly.                                 */
#   define FP_FAST_FMA     1
#   define FP_FAST_FMAF    1
#   undef  FP_FAST_FMAL
#else
/*  On these architectures, fma( ), fmaf( ), and fmal( ) function calls are
    significantly more costly than separate multiply and add operations.      */
#   undef  FP_FAST_FMA
#   undef  FP_FAST_FMAF
#   undef  FP_FAST_FMAL
#endif

/* The values returned by `ilogb' for 0 and NaN respectively. */
#define FP_ILOGB0      (-2147483647 - 1)
#define FP_ILOGBNAN    (-2147483647 - 1)

/* Bitmasks for the math_errhandling macro.  */
#define MATH_ERRNO        1    /* errno set by math functions.  */
#define MATH_ERREXCEPT    2    /* Exceptions raised by math functions.  */

#define math_errhandling (__math_errhandling())
extern int __math_errhandling(void);

/******************************************************************************
 *                                                                            *
 *                              Inquiry macros                                *
 *                                                                            *
 *  fpclassify      Returns one of the FP_* values.                           *
 *  isnormal        Non-zero if and only if the argument x is normalized.     *
 *  isfinite        Non-zero if and only if the argument x is finite.         *
 *  isnan           Non-zero if and only if the argument x is a NaN.          *
 *  signbit         Non-zero if and only if the sign of the argument x is     *
 *                  negative.  This includes, NaNs, infinities and zeros.     *
 *                                                                            *
 ******************************************************************************/

#if (defined __MAC_OS_X_VERSION_MIN_REQUIRED && __MAC_OS_X_VERSION_MIN_REQUIRED < 1080) || \
    (defined __IPHONE_OS_VERSION_MIN_REQUIRED && __IPHONE_OS_VERSION_MIN_REQUIRED < 60000)
#   if defined __i386__ || defined __x86_64__
#       define __fpclassifyl __fpclassify
#       define __isnormall   __isnormal
#       define __isfinitel   __isfinite
#       define __isinfl      __isinf
#       define __isnanl      __isnan
#   elif defined __arm__
#       define __fpclassifyd __fpclassify
#   endif
#endif

#define fpclassify(x)                                                    \
    ( sizeof(x) == sizeof(float)  ? __fpclassifyf((float)(x))            \
    : sizeof(x) == sizeof(double) ? __fpclassifyd((double)(x))           \
                                  : __fpclassifyl((long double)(x)))

extern int __fpclassifyf(float);
extern int __fpclassifyd(double);
extern int __fpclassifyl(long double);

/*  Implementations making function calls to fall back on when -ffast-math
    or similar is specified.  These are not available in iOS versions prior
    to 6.0.  If you need them, you must target that version or later.         */
    
#define isnormal(x)                                               \
    ( sizeof(x) == sizeof(float)  ? __isnormalf((float)(x))       \
    : sizeof(x) == sizeof(double) ? __isnormald((double)(x))      \
                                  : __isnormall((long double)(x)))
    
#define isfinite(x)                                               \
    ( sizeof(x) == sizeof(float)  ? __isfinitef((float)(x))       \
    : sizeof(x) == sizeof(double) ? __isfinited((double)(x))      \
                                  : __isfinitel((long double)(x)))
    
#define isinf(x)                                                  \
    ( sizeof(x) == sizeof(float)  ? __isinff((float)(x))          \
    : sizeof(x) == sizeof(double) ? __isinfd((double)(x))         \
                                  : __isinfl((long double)(x)))
    
#define isnan(x)                                                  \
    ( sizeof(x) == sizeof(float)  ? __isnanf((float)(x))          \
    : sizeof(x) == sizeof(double) ? __isnand((double)(x))         \
                                  : __isnanl((long double)(x)))
    
#define signbit(x)                                                \
    ( sizeof(x) == sizeof(float)  ? __signbitf((float)(x))        \
    : sizeof(x) == sizeof(double) ? __signbitd((double)(x))       \
                                  : __signbitl((long double)(x)))
    
extern int __isnormalf(float);
extern int __isnormald(double);
extern int __isnormall(long double);
extern int __isfinitef(float);
extern int __isfinited(double);
extern int __isfinitel(long double);
extern int __isinff(float);
extern int __isinfd(double);
extern int __isinfl(long double);
extern int __isnanf(float);
extern int __isnand(double);
extern int __isnanl(long double);
extern int __signbitf(float);
extern int __signbitd(double);
extern int __signbitl(long double);

/******************************************************************************
 *                                                                            *
 *                              Math Functions                                *
 *                                                                            *
 ******************************************************************************/
    
extern float acosf(float);
extern double acos(double);
extern long double acosl(long double);
    
extern float asinf(float);
extern double asin(double);
extern long double asinl(long double);
    
extern float atanf(float);
extern double atan(double);
extern long double atanl(long double);
    
extern float atan2f(float, float);
extern double atan2(double, double);
extern long double atan2l(long double, long double);
    
extern float cosf(float);
extern double cos(double);
extern long double cosl(long double);
    
extern float sinf(float);
extern double sin(double);
extern long double sinl(long double);
    
extern float tanf(float);
extern double tan(double);
extern long double tanl(long double);
    
extern float acoshf(float);
extern double acosh(double);
extern long double acoshl(long double);
    
extern float asinhf(float);
extern double asinh(double);
extern long double asinhl(long double);
    
extern float atanhf(float);
extern double atanh(double);
extern long double atanhl(long double);
    
extern float coshf(float);
extern double cosh(double);
extern long double coshl(long double);
    
extern float sinhf(float);
extern double sinh(double);
extern long double sinhl(long double);
    
extern float tanhf(float);
extern double tanh(double);
extern long double tanhl(long double);
    
extern float expf(float);
extern double exp(double);
extern long double expl(long double);

extern float exp2f(float);
extern double exp2(double); 
extern long double exp2l(long double); 

extern float expm1f(float);
extern double expm1(double); 
extern long double expm1l(long double); 

extern float logf(float);
extern double log(double);
extern long double logl(long double);

extern float log10f(float);
extern double log10(double);
extern long double log10l(long double);

extern float log2f(float);
extern double log2(double);
extern long double log2l(long double);

extern float log1pf(float);
extern double log1p(double);
extern long double log1pl(long double);

extern float logbf(float);
extern double logb(double);
extern long double logbl(long double);

extern float modff(float, float *);
extern double modf(double, double *);
extern long double modfl(long double, long double *);

extern float ldexpf(float, int);
extern double ldexp(double, int);
extern long double ldexpl(long double, int);

extern float frexpf(float, int *);
extern double frexp(double, int *);
extern long double frexpl(long double, int *);

extern int ilogbf(float);
extern int ilogb(double);
extern int ilogbl(long double);

extern float scalbnf(float, int);
extern double scalbn(double, int);
extern long double scalbnl(long double, int);

extern float scalblnf(float, long int);
extern double scalbln(double, long int);
extern long double scalblnl(long double, long int);

extern float fabsf(float);
extern double fabs(double);
extern long double fabsl(long double);

extern float cbrtf(float);
extern double cbrt(double);
extern long double cbrtl(long double);

extern float hypotf(float, float);
extern double hypot(double, double);
extern long double hypotl(long double, long double);

extern float powf(float, float);
extern double pow(double, double);
extern long double powl(long double, long double);

extern float sqrtf(float);
extern double sqrt(double);
extern long double sqrtl(long double);

extern float erff(float);
extern double erf(double);
extern long double erfl(long double);

extern float erfcf(float);
extern double erfc(double);
extern long double erfcl(long double);

/*	lgammaf, lgamma, and lgammal are not thread-safe. The thread-safe
    variants lgammaf_r, lgamma_r, and lgammal_r are made available if
    you define the _REENTRANT symbol before including <math.h>                */
extern float lgammaf(float);
extern double lgamma(double);
extern long double lgammal(long double);

extern float tgammaf(float);
extern double tgamma(double);
extern long double tgammal(long double);

extern float ceilf(float);
extern double ceil(double);
extern long double ceill(long double);

extern float floorf(float);
extern double floor(double);
extern long double floorl(long double);

extern float nearbyintf(float);
extern double nearbyint(double);
extern long double nearbyintl(long double);

extern float rintf(float);
extern double rint(double);
extern long double rintl(long double);

extern long int lrintf(float);
extern long int lrint(double);
extern long int lrintl(long double);

extern float roundf(float);
extern double round(double);
extern long double roundl(long double);

extern long int lroundf(float);
extern long int lround(double);
extern long int lroundl(long double);
    
extern long long int llrintf(float);
extern long long int llrint(double);
extern long long int llrintl(long double);

extern long long int llroundf(float);
extern long long int llround(double);
extern long long int llroundl(long double);

extern float truncf(float);
extern double trunc(double);
extern long double truncl(long double);

extern float fmodf(float, float);
extern double fmod(double, double);
extern long double fmodl(long double, long double);

extern float remainderf(float, float);
extern double remainder(double, double);
extern long double remainderl(long double, long double);

extern float remquof(float, float, int *);
extern double remquo(double, double, int *);
extern long double remquol(long double, long double, int *);

extern float copysignf(float, float);
extern double copysign(double, double);
extern long double copysignl(long double, long double);

extern float nanf(const char *);
extern double nan(const char *);
extern long double nanl(const char *);

extern float nextafterf(float, float);
extern double nextafter(double, double);
extern long double nextafterl(long double, long double);

extern double nexttoward(double, long double);
extern float nexttowardf(float, long double);
extern long double nexttowardl(long double, long double);

extern float fdimf(float, float);
extern double fdim(double, double);
extern long double fdiml(long double, long double);

extern float fmaxf(float, float);
extern double fmax(double, double);
extern long double fmaxl(long double, long double);

extern float fminf(float, float);
extern double fmin(double, double);
extern long double fminl(long double, long double);

extern float fmaf(float, float, float);
extern double fma(double, double, double);
extern long double fmal(long double, long double, long double);

#define isgreater(x, y) __builtin_isgreater((x),(y))
#define isgreaterequal(x, y) __builtin_isgreaterequal((x),(y))
#define isless(x, y) __builtin_isless((x),(y))
#define islessequal(x, y) __builtin_islessequal((x),(y))
#define islessgreater(x, y) __builtin_islessgreater((x),(y))
#define isunordered(x, y) __builtin_isunordered((x),(y))

