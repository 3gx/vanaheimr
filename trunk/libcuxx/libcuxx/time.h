
#pragma once

struct tm
{
	int tm_sec;   // Seconds [0,60]. 
	int tm_min;   // Minutes [0,59]. 
	int tm_hour;  // Hour [0,23]. 
	int tm_mday;  // Day of month [1,31]. 
	int tm_mon;   // Month of year [0,11]. 
	int tm_year;  // Years since 1900. 
	int tm_wday;  // Day of week [0,6] (Sunday =0). 
	int tm_yday;  // Day of year [0,365]. 
	int tm_isdst; // Daylight Savings flag. 

};

typedef long clock_t;
typedef long time_t;
typedef long timer_t;
typedef long clockid_t;

struct timespec
{
	time_t  tv_sec;   // Seconds. 
	long    tv_nsec;  // Nanoseconds. 

};

extern char      *asctime(const struct tm *);
extern char      *asctime_r(const struct tm *, char *);
extern clock_t    clock(void);
extern int        clock_getres(clockid_t, struct timespec *);
extern int        clock_gettime(clockid_t, struct timespec *);
extern int        clock_settime(clockid_t, const struct timespec *);
extern char      *ctime(const time_t *);
extern char      *ctime_r(const time_t *, char *);
extern double     difftime(time_t, time_t);
extern struct tm *getdate(const char *);
extern struct tm *gmtime(const time_t *);
extern struct tm *gmtime_r(const time_t *, struct tm *);
extern struct tm *localtime(const time_t *);
extern struct tm *localtime_r(const time_t *, struct tm *);
extern time_t     mktime(struct tm *);
extern int        nanosleep(const struct timespec *, struct timespec *);
extern size_t     strftime(char *, size_t, const char *, const struct tm *);
extern char      *strptime(const char *, const char *, struct tm *);
extern time_t     time(time_t *);
extern int        timer_create(clockid_t, struct sigevent *, timer_t *);
extern int        timer_delete(timer_t);
extern int        timer_gettime(timer_t, struct itimerspec *);
extern int        timer_getoverrun(timer_t);
extern int        timer_settime(timer_t, int, const struct itimerspec *,
               struct itimerspec *);
extern void       tzset(void);


