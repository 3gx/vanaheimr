#pragma once

#include <string.h>
#include <wchar.h>

struct lconv
{
	char    *currency_symbol;
	char    *decimal_point;
	char     frac_digits;
	char    *grouping;
	char    *int_curr_symbol;
	char     int_frac_digits;
	char     int_n_cs_precedes;
	char     int_n_sep_by_space;
	char     int_n_sign_posn;
	char     int_p_cs_precedes;
	char     int_p_sep_by_space;
	char     int_p_sign_posn;
	char    *mon_decimal_point;
	char    *mon_grouping;
	char    *mon_thousands_sep;
	char    *negative_sign;
	char     n_cs_precedes;
	char     n_sep_by_space;
	char     n_sign_posn;
	char    *positive_sign;
	char     p_cs_precedes;
	char     p_sep_by_space;
	char     p_sign_posn;
	char    *thousands_sep;
};

typedef int* locale_t;

extern locale_t newlocale(int category_mask, const char *locale,
       locale_t base);
extern void freelocale(locale_t locobj);

/* Set and/or return the current locale.  */
extern char *uselocale (int __category, const char *__locale);
extern char *setlocale (int __category, const char *__locale);
//extern char *freelocale (int __category, const char *__locale);

/* Return the numeric/monetary information for the current locale.  */
extern struct lconv *localeconv (void);
extern struct lconv *__cloc();
extern struct lconv *localeconv_l(locale_t loc);

extern long strtol_l(const char * nptr, char ** endptr, int base, locale_t loc);
extern long long strtoll_l(const char * nptr, char ** endptr, int base,	locale_t loc);
extern unsigned long strtoul_l(const char * nptr, char ** endptr, int base,	locale_t loc);
extern unsigned long long strtoull_l(const char * nptr, char ** endptr, int base, locale_t loc);
extern double strtod_l(const char * nptr, char ** endptr, locale_t loc);
extern float strtof_l(const char * nptr, char ** endptr, locale_t loc);
extern long double strtold_l(const char * nptr, char ** endptr, locale_t loc);
extern int strcoll_l ( const char * str1, const char * str2, locale_t loc );
extern size_t strxfrm_l(char *, const char *, size_t, locale_t);

extern int isupper_l(int character, locale_t);
extern int islower_l(int character, locale_t);
extern int iswupper_l(int character, locale_t);
extern int iswlower_l(int character, locale_t);

extern int iswspace_l(wint_t, locale_t);
extern int isspace_l(int, locale_t);

extern int iswprint_l(wint_t, locale_t);

extern int tolower_l(int character, locale_t);
extern int toupper_l(int character, locale_t);
extern int towlower_l(int character, locale_t);
extern int towupper_l(int character, locale_t);

extern int isxdigit_l(int character, locale_t loc);
extern int isdigit_l(int character, locale_t loc);

extern int iswalnum_l(wint_t, locale_t);
extern int iswalpha_l(wint_t, locale_t);
extern int iswcntrl_l(wint_t, locale_t);
extern int iswctype_l(wint_t, wctype_t, locale_t);
extern int iswdigit_l(wint_t, locale_t);
extern int iswgraph_l(wint_t, locale_t);
extern int iswpunct_l(wint_t, locale_t);
extern int iswxdigit_l(wint_t, locale_t);
extern int iswblank_l(wint_t, locale_t);

extern int wcscoll_l(const wchar_t *, const wchar_t *, locale_t);
extern size_t wcsxfrm_l(wchar_t *, const wchar_t *, size_t, locale_t);
extern size_t wcrtomb_l(char * s, wchar_t wc, mbstate_t * ps, locale_t loc);

extern wint_t btowc_l(int, locale_t);
extern int    wctob_l(wint_t, locale_t);

extern int sscanf_l(const char *, locale_t, const char *, ...);
extern int snprintf_l(char *, size_t, locale_t, const char *, ...);
extern int asprintf_l(char **strp, locale_t, const char *fmt, ...);

extern size_t strftime_l(char * s, size_t maxsize, const char * format, const struct tm * timeptr, locale_t locale);

extern size_t mbsrtowcs_l(wchar_t * dst, const char ** src, size_t len, mbstate_t * ps, locale_t loc);
extern size_t wcsnrtombs_l(char * dst, const wchar_t ** src, size_t nwc, size_t len, mbstate_t * ps, locale_t loc);
extern size_t mbsnrtowcs_l(wchar_t * dst, const char ** src, size_t nms, size_t len, mbstate_t * ps, locale_t loc);
extern size_t mbrtowc_l(wchar_t * pwc, const char * s, size_t n, mbstate_t * ps, locale_t loc);
extern int mbtowc_l(wchar_t * pwc, const char * s, size_t n, locale_t loc);
extern size_t mbrlen_l(const char * s, size_t n, mbstate_t * ps, locale_t loc);

extern size_t __mb_cur_max_l(locale_t l);

#define MB_CUR_MAX_L(x) (__mb_cur_max_l(x))


#define LC_CTYPE          1
#define LC_NUMERIC        2
#define LC_TIME           3
#define LC_COLLATE        4
#define LC_MONETARY       5
#define LC_MESSAGES       6

#define LC_PAPER          7
#define LC_NAME           8
#define LC_ADDRESS        9
#define LC_TELEPHONE      10
#define LC_MEASUREMENT    11
#define LC_IDENTIFICATION 12

# define LC_CTYPE_MASK		(1 << LC_CTYPE)
# define LC_NUMERIC_MASK	(1 << LC_NUMERIC)
# define LC_TIME_MASK		(1 << LC_TIME)
# define LC_COLLATE_MASK	(1 << LC_COLLATE)
# define LC_MONETARY_MASK	(1 << LC_MONETARY)
# define LC_MESSAGES_MASK	(1 << LC_MESSAGES)

# define LC_PAPER_MASK		(1 << LC_PAPER)
# define LC_NAME_MASK		(1 << LC_NAME)
# define LC_ADDRESS_MASK	(1 << LC_ADDRESS)
# define LC_TELEPHONE_MASK	(1 << LC_TELEPHONE)
# define LC_MEASUREMENT_MASK	(1 << LC_MEASUREMENT)
# define LC_IDENTIFICATION_MASK	(1 << LC_IDENTIFICATION)
# define LC_ALL_MASK		(LC_CTYPE_MASK \
				 | LC_NUMERIC_MASK \
				 | LC_TIME_MASK \
				 | LC_COLLATE_MASK \
				 | LC_MONETARY_MASK \
				 | LC_MESSAGES_MASK \
				 | LC_PAPER_MASK \
				 | LC_NAME_MASK \
				 | LC_ADDRESS_MASK \
				 | LC_TELEPHONE_MASK \
				 | LC_MEASUREMENT_MASK \
				 | LC_IDENTIFICATION_MASK)

