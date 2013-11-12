
#pragma once

void* memcpy (void* destination, const void* source, size_t num);

extern char * strcpy(char *,const char *);
extern char * strncpy(char *,const char *, size_t);
size_t strlcpy(char *, const char *, size_t);
extern char * strcat(char *, const char *);
extern char * strncat(char *, const char *, size_t);
extern size_t strlcat(char *, const char *, size_t);
extern int strcmp(const char *,const char *);
extern int strncmp(const char *,const char *,size_t);
extern int strnicmp(const char *, const char *, size_t);
extern int strcasecmp(const char *s1, const char *s2);
extern int strncasecmp(const char *s1, const char *s2, size_t n);
extern char * strchr(const char *,int);
extern char * strnchr(const char *, size_t, int);
extern char * strrchr(const char *,int);
extern char * skip_spaces(const char *);
extern char *strim(char *);

static inline char *strstrip(char *str)
{
	return strim(str);
}

extern char * strstr(const char *, const char *);
extern char * strnstr(const char *, const char *, size_t);
extern size_t strlen(const char *);
extern size_t strnlen(const char *,size_t);
extern char * strpbrk(const char *,const char *);
extern char * strsep(char **,const char *);
extern size_t strspn(const char *,const char *);
extern size_t strcspn(const char *,const char *);
extern void * memset(void *,int,size_t);
extern void * memcpy(void *,const void *,size_t);
extern void * memmove(void *,const void *,size_t);
extern void * memscan(void *,int,size_t);
extern int memcmp(const void *,const void *,size_t);
extern void * memchr(const void *,int,size_t);
void *memchr_inv(const void *s, int c, size_t n);

