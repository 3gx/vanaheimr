
#include <locale.h>
#include <assert.h>

extern locale_t newlocale(int category_mask, const char *locale,
       locale_t base)
{
	assert(false && "not implemented");

	return 0;
}

extern void freelocale(locale_t locobj)
{
	assert(false && "not implemented");
}

extern char *uselocale (int __category, const char *__locale)
{
	assert(false && "not implemented");

	return 0;
}

extern char *setlocale (int __category, const char *__locale)
{
	assert(false && "not implemented");

	return 0;
}

extern struct lconv *localeconv (void)
{
	assert(false && "not implemented");

	return 0;
}

extern struct lconv *__cloc()
{
	assert(false && "not implemented");

	return 0;
}

extern struct lconv *localeconv_l(locale_t loc)
{
	assert(false && "not implemented");

	return 0;
}

extern long strtol_l(const char * nptr, char ** endptr, int base, locale_t loc)
{
	assert(false && "not implemented");

	return 0;
}

extern long long strtoll_l(const char * nptr, char ** endptr, int base,	locale_t loc)
{
	assert(false && "not implemented");

	return 0;
}

extern unsigned long strtoul_l(const char * nptr, char ** endptr, int base,	locale_t loc)
{
	assert(false && "not implemented");

	return 0;
}

extern unsigned long long strtoull_l(const char * nptr, char ** endptr, int base, locale_t loc)
{
	assert(false && "not implemented");

	return 0;
}

extern double strtod_l(const char * nptr, char ** endptr, locale_t loc)
{
	assert(false && "not implemented");

	return 0;
}

extern float strtof_l(const char * nptr, char ** endptr, locale_t loc)
{
	assert(false && "not implemented");

	return 0;
}

extern long double strtold_l(const char * nptr, char ** endptr, locale_t loc)
{
	assert(false && "not implemented");

	return 0;
}

extern int strcoll_l ( const char * str1, const char * str2, locale_t loc )
{
	assert(false && "not implemented");

	return 0;
}

extern size_t strxfrm_l(char *, const char *, size_t, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int isupper_l(int character, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int islower_l(int character, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int iswupper_l(int character, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int iswlower_l(int character, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int iswspace_l(wint_t, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int isspace_l(int, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int iswprint_l(wint_t, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int tolower_l(int character, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int toupper_l(int character, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int towlower_l(int character, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int towupper_l(int character, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int isxdigit_l(int character, locale_t loc)
{
	assert(false && "not implemented");

	return 0;
}

extern int isdigit_l(int character, locale_t loc)
{
	assert(false && "not implemented");

	return 0;
}

extern int iswalnum_l(wint_t, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int iswalpha_l(wint_t, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int iswcntrl_l(wint_t, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int iswctype_l(wint_t, wctype_t, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int iswdigit_l(wint_t, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int iswgraph_l(wint_t, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int iswpunct_l(wint_t, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int iswxdigit_l(wint_t, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int iswblank_l(wint_t, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int wcscoll_l(const wchar_t *, const wchar_t *, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern size_t wcsxfrm_l(wchar_t *, const wchar_t *, size_t, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern size_t wcrtomb_l(char * s, wchar_t wc, mbstate_t * ps, locale_t loc)
{
	assert(false && "not implemented");

	return 0;
}

extern wint_t btowc_l(int, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int    wctob_l(wint_t, locale_t)
{
	assert(false && "not implemented");

	return 0;
}

extern int sscanf_l(const char *, locale_t, const char *, ...)
{
	assert(false && "not implemented");

	return 0;
}

extern int snprintf_l(char *, size_t, locale_t, const char *, ...)
{
	assert(false && "not implemented");

	return 0;
}

extern int asprintf_l(char **strp, locale_t, const char *fmt, ...)
{
	assert(false && "not implemented");

	return 0;
}

extern size_t strftime_l(char * s, size_t maxsize, const char * format, const struct tm * timeptr, locale_t locale)
{
	assert(false && "not implemented");

	return 0;
}

extern size_t mbsrtowcs_l(wchar_t * dst, const char ** src, size_t len, mbstate_t * ps, locale_t loc)
{
	assert(false && "not implemented");

	return 0;
}

extern size_t wcsnrtombs_l(char * dst, const wchar_t ** src, size_t nwc, size_t len, mbstate_t * ps, locale_t loc)
{
	assert(false && "not implemented");

	return 0;
}

extern size_t mbsnrtowcs_l(wchar_t * dst, const char ** src, size_t nms, size_t len, mbstate_t * ps, locale_t loc)
{
	assert(false && "not implemented");

	return 0;
}

extern size_t mbrtowc_l(wchar_t * pwc, const char * s, size_t n, mbstate_t * ps, locale_t loc)
{
	assert(false && "not implemented");

	return 0;
}

extern int mbtowc_l(wchar_t * pwc, const char * s, size_t n, locale_t loc)
{
	assert(false && "not implemented");

	return 0;
}

extern size_t mbrlen_l(const char * s, size_t n, mbstate_t * ps, locale_t loc)
{
	assert(false && "not implemented");

	return 0;
}

extern size_t __mb_cur_max_l(locale_t l)
{
	assert(false && "not implemented");

	return 0;
}


