#pragma once

typedef int nl_catd;
typedef int nl_item;

#define NL_SETD 0
#define NL_CAT_LOCALE 0

extern int       catclose(nl_catd);
extern char     *catgets(nl_catd, int, int, const char *);
extern nl_catd   catopen(const char *, int);


