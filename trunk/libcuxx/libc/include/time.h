
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
typedef long suseconds_t;

struct timespec
{
	time_t  tv_sec;   // Seconds. 
	long    tv_nsec;  // Nanoseconds. 

};

struct timeval
{
        time_t      tv_sec;         /* seconds */
        suseconds_t tv_usec;        /* and microseconds */
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
extern int   gettimeofday(struct timeval *, void *);


enum
{
CLOCK_REALTIME,
	  /*System-wide clock that measures real (i.e., wall-clock) time.
	  Setting this clock requires appropriate privileges.  This
	  clock is affected by discontinuous jumps in the system time
	  (e.g., if the system administrator manually changes the
	  clock), and by the incremental adjustments performed by
	  adjtime(3) and NTP.*/

CLOCK_REALTIME_COARSE,
	/*
	 (since Linux 2.6.32; Linux-specific)
	  A faster but less precise version of CLOCK_REALTIME.  Use when
	  you need very fast, but not fine-grained timestamps. */

CLOCK_MONOTONIC,
		/*
	  Clock that cannot be set and represents monotonic time
	  since some unspecified starting point.  This clock is
		  not affected by discontinuous jumps in the system time
		  (e.g., if the system administrator manually changes the
		  clock), but is affected by the incremental adjustments
		  performed by adjtime(3) and NTP.*/

   CLOCK_MONOTONIC_COARSE, /*(since Linux 2.6.32; Linux-specific)
		  A faster but less precise version of CLOCK_MONOTONIC.
		  Use when you need very fast, but not fine-grained
		  timestamps.*/

   CLOCK_MONOTONIC_RAW, /*(since Linux 2.6.28; Linux-specific)
		  Similar to CLOCK_MONOTONIC, but provides access to a
		  raw hardware-based time that is not subject to NTP
		  adjustments or the incremental adjustments performed by
		  adjtime(3).*/

   CLOCK_BOOTTIME, /*(since Linux 2.6.39; Linux-specific)
		  Identical to CLOCK_MONOTONIC, except it also includes
		  any time that the system is suspended.  This allows
		  applications to get a suspend-aware monotonic clock
		  without having to deal with the complications of
		  CLOCK_REALTIME, which may have discontinuities if the
		  time is changed using settimeofday(2).*/

   CLOCK_PROCESS_CPUTIME_ID,/* (since Linux 2.6.12)
		  High-resolution per-process timer from the CPU. */

   CLOCK_THREAD_CPUTIME_ID /*(since Linux 2.6.12)
		  Thread-specific CPU-time clock.*/

};


