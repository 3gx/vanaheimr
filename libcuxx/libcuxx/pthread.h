
#pragma once

#include <sched.h>

// Macros


// PTHREAD_CANCEL_ASYNCHRONOUS
// PTHREAD_CANCEL_ENABLE
// PTHREAD_CANCEL_DEFERRED
// PTHREAD_CANCEL_DISABLE
enum
{
  PTHREAD_CANCEL_ENABLE,
#define PTHREAD_CANCEL_ENABLE   PTHREAD_CANCEL_ENABLE
  PTHREAD_CANCEL_DISABLE
#define PTHREAD_CANCEL_DISABLE  PTHREAD_CANCEL_DISABLE
};
enum
{
  PTHREAD_CANCEL_DEFERRED,
#define PTHREAD_CANCEL_DEFERRED	PTHREAD_CANCEL_DEFERRED
  PTHREAD_CANCEL_ASYNCHRONOUS
#define PTHREAD_CANCEL_ASYNCHRONOUS	PTHREAD_CANCEL_ASYNCHRONOUS
};

// PTHREAD_CANCELED
#define PTHREAD_CANCELED ((void *) -1)

// PTHREAD_COND_INITIALIZER
#define PTHREAD_COND_INITIALIZER { { 0, 0 }, 0, 0 }


// PTHREAD_CREATE_DETACHED
// PTHREAD_CREATE_JOINABLE
enum
{
  PTHREAD_CREATE_JOINABLE,
#define PTHREAD_CREATE_JOINABLE	PTHREAD_CREATE_JOINABLE
  PTHREAD_CREATE_DETACHED
#define PTHREAD_CREATE_DETACHED	PTHREAD_CREATE_DETACHED
};

// PTHREAD_EXPLICIT_SCHED
// PTHREAD_INHERIT_SCHED
enum
{
  PTHREAD_INHERIT_SCHED,
#define PTHREAD_INHERIT_SCHED   PTHREAD_INHERIT_SCHED
  PTHREAD_EXPLICIT_SCHED
#define PTHREAD_EXPLICIT_SCHED  PTHREAD_EXPLICIT_SCHED
};

// PTHREAD_MUTEX_DEFAULT
// PTHREAD_MUTEX_ERRORCHECK
// PTHREAD_MUTEX_NORMAL
// PTHREAD_MUTEX_RECURSIVE
enum
{
  PTHREAD_MUTEX_TIMED_NP,
  PTHREAD_MUTEX_RECURSIVE_NP,
  PTHREAD_MUTEX_ERRORCHECK_NP,
  PTHREAD_MUTEX_ADAPTIVE_NP,
  PTHREAD_MUTEX_NORMAL = PTHREAD_MUTEX_TIMED_NP,
  PTHREAD_MUTEX_RECURSIVE = PTHREAD_MUTEX_RECURSIVE_NP,
  PTHREAD_MUTEX_ERRORCHECK = PTHREAD_MUTEX_ERRORCHECK_NP,
  PTHREAD_MUTEX_DEFAULT = PTHREAD_MUTEX_NORMAL
};

// PTHREAD_MUTEX_INITIALIZER
# define PTHREAD_MUTEX_INITIALIZER \
  { 0, 0, 0, 0, { 0, 0 } }

// PTHREAD_ONCE_INIT
#define PTHREAD_ONCE_INIT 0

// PTHREAD_PRIO_INHERIT
// PTHREAD_PRIO_NONE
// PTHREAD_PRIO_PROTECT
enum
{
  PTHREAD_PRIO_NONE,
  PTHREAD_PRIO_INHERIT,
  PTHREAD_PRIO_PROTECT
};

// PTHREAD_PROCESS_SHARED
// PTHREAD_PROCESS_PRIVATE
enum
{
  PTHREAD_PROCESS_PRIVATE,
#define PTHREAD_PROCESS_PRIVATE PTHREAD_PROCESS_PRIVATE
  PTHREAD_PROCESS_SHARED
#define PTHREAD_PROCESS_SHARED  PTHREAD_PROCESS_SHARED
};

// PTHREAD_RWLOCK_INITIALIZER
# define PTHREAD_RWLOCK_INITIALIZER \
  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }

// PTHREAD_SCOPE_PROCESS
// PTHREAD_SCOPE_SYSTEM
enum
{
  PTHREAD_SCOPE_SYSTEM,
#define PTHREAD_SCOPE_SYSTEM    PTHREAD_SCOPE_SYSTEM
  PTHREAD_SCOPE_PROCESS
#define PTHREAD_SCOPE_PROCESS   PTHREAD_SCOPE_PROCESS
};


// types

struct _pthread_cleanup_buffer {
    void (*__routine) (void *);
    void *__arg;
    int __canceltype;
    struct _pthread_cleanup_buffer *__prev;
};

typedef unsigned int pthread_key_t;
typedef int pthread_once_t;
typedef long long int __pthread_cond_align_t;
struct sched_param {};

typedef unsigned long int pthread_t;

struct _pthread_fastlock {
    long int __status;
    int __spinlock;
};

typedef struct _pthread_descr_struct *_pthread_descr;

typedef struct {
    int __m_reserved;
    int __m_count;
    _pthread_descr __m_owner;
    int __m_kind;
    struct _pthread_fastlock __m_lock;
} pthread_mutex_t;

typedef struct {
    int __mutexkind;
} pthread_mutexattr_t;

typedef struct {
    int __detachstate;
    int __schedpolicy;
    struct sched_param __schedparam;
    int __inheritsched;
    int __scope;
    size_t __guardsize;
    int __stackaddr_set;
    void *__stackaddr;
    unsigned long int __stacksize;
} pthread_attr_t;

typedef struct {
    struct _pthread_fastlock __c_lock;
    _pthread_descr __c_waiting;
    char __padding[48 - sizeof(struct _pthread_fastlock) -
		   sizeof(_pthread_descr) -
		   sizeof(__pthread_cond_align_t)];
    __pthread_cond_align_t __align;
} pthread_cond_t;

typedef struct {
    int __dummy;
} pthread_condattr_t;

typedef struct _pthread_rwlock_t {
    struct _pthread_fastlock __rw_lock;
    int __rw_readers;
    _pthread_descr __rw_writer;
    _pthread_descr __rw_read_waiting;
    _pthread_descr __rw_write_waiting;
    int __rw_kind;
    int __rw_pshared;
} pthread_rwlock_t;

typedef struct {
    int __lockkind;
    int __pshared;
} pthread_rwlockattr_t;

// Functions

extern int   pthread_attr_destroy(pthread_attr_t *);
extern int   pthread_attr_getdetachstate(const pthread_attr_t *, int *);
extern int   pthread_attr_getguardsize(const pthread_attr_t *, size_t *);
extern int   pthread_attr_getinheritsched(const pthread_attr_t *, int *);
extern int   pthread_attr_getschedparam(const pthread_attr_t *,
          struct sched_param *);
extern int   pthread_attr_getschedpolicy(const pthread_attr_t *, int *);
extern int   pthread_attr_getscope(const pthread_attr_t *, int *);
extern int   pthread_attr_getstackaddr(const pthread_attr_t *, void **);
extern int   pthread_attr_getstacksize(const pthread_attr_t *, size_t *);
extern int   pthread_attr_init(pthread_attr_t *);
extern int   pthread_attr_setdetachstate(pthread_attr_t *, int);
extern int   pthread_attr_setguardsize(pthread_attr_t *, size_t);
extern int   pthread_attr_setinheritsched(pthread_attr_t *, int);
extern int   pthread_attr_setschedparam(pthread_attr_t *,
          const struct sched_param *);
extern int   pthread_attr_setschedpolicy(pthread_attr_t *, int);
extern int   pthread_attr_setscope(pthread_attr_t *, int);
extern int   pthread_attr_setstackaddr(pthread_attr_t *, void *);
extern int   pthread_attr_setstacksize(pthread_attr_t *, size_t);
extern int   pthread_cancel(pthread_t);
extern void  pthread_cleanup_push(void (*routine)(void *), void *);
extern void  pthread_cleanup_pop(int);
extern int   pthread_cond_broadcast(pthread_cond_t *);
extern int   pthread_cond_destroy(pthread_cond_t *);
extern int   pthread_cond_init(pthread_cond_t *, const pthread_condattr_t *);
extern int   pthread_cond_signal(pthread_cond_t *);
extern int   pthread_cond_timedwait(pthread_cond_t *, 
          pthread_mutex_t *, const struct timespec *);
extern int   pthread_cond_wait(pthread_cond_t *, pthread_mutex_t *);
extern int   pthread_condattr_destroy(pthread_condattr_t *);
extern int   pthread_condattr_getpshared(const pthread_condattr_t *, int *);
extern int   pthread_condattr_init(pthread_condattr_t *);
extern int   pthread_condattr_setpshared(pthread_condattr_t *, int);
extern int   pthread_create(pthread_t *, const pthread_attr_t *,
          void *(*)(void *), void *);
extern int   pthread_detach(pthread_t);
extern int   pthread_equal(pthread_t, pthread_t);
extern void  pthread_exit(void *);
extern int   pthread_getconcurrency(void);
extern int   pthread_getschedparam(pthread_t, int *, struct sched_param *);
extern void *pthread_getspecific(pthread_key_t);
extern int   pthread_join(pthread_t, void **);
extern int   pthread_key_create(pthread_key_t *, void (*)(void *));
extern int   pthread_key_delete(pthread_key_t);
extern int   pthread_mutex_destroy(pthread_mutex_t *);
extern int   pthread_mutex_getprioceiling(const pthread_mutex_t *, int *);
extern int   pthread_mutex_init(pthread_mutex_t *, const pthread_mutexattr_t *);
extern int   pthread_mutex_lock(pthread_mutex_t *);
extern int   pthread_mutex_setprioceiling(pthread_mutex_t *, int, int *);
extern int   pthread_mutex_trylock(pthread_mutex_t *);
extern int   pthread_mutex_unlock(pthread_mutex_t *);
extern int   pthread_mutexattr_destroy(pthread_mutexattr_t *);
extern int   pthread_mutexattr_getprioceiling(const pthread_mutexattr_t *,
          int *);
extern int   pthread_mutexattr_getprotocol(const pthread_mutexattr_t *, int *);
extern int   pthread_mutexattr_getpshared(const pthread_mutexattr_t *, int *);
extern int   pthread_mutexattr_gettype(const pthread_mutexattr_t *, int *);
extern int   pthread_mutexattr_init(pthread_mutexattr_t *);
extern int   pthread_mutexattr_setprioceiling(pthread_mutexattr_t *, int);
extern int   pthread_mutexattr_setprotocol(pthread_mutexattr_t *, int);
extern int   pthread_mutexattr_setpshared(pthread_mutexattr_t *, int);
extern int   pthread_mutexattr_settype(pthread_mutexattr_t *, int);
extern int   pthread_once(pthread_once_t *, void (*)(void));
extern int   pthread_rwlock_destroy(pthread_rwlock_t *);
extern int   pthread_rwlock_init(pthread_rwlock_t *,
          const pthread_rwlockattr_t *);
extern int   pthread_rwlock_rdlock(pthread_rwlock_t *);
extern int   pthread_rwlock_tryrdlock(pthread_rwlock_t *);
extern int   pthread_rwlock_trywrlock(pthread_rwlock_t *);
extern int   pthread_rwlock_unlock(pthread_rwlock_t *);
extern int   pthread_rwlock_wrlock(pthread_rwlock_t *);
extern int   pthread_rwlockattr_destroy(pthread_rwlockattr_t *);
extern int   pthread_rwlockattr_getpshared(const pthread_rwlockattr_t *,
          int *);
extern int   pthread_rwlockattr_init(pthread_rwlockattr_t *);
extern int   pthread_rwlockattr_setpshared(pthread_rwlockattr_t *, int);
extern pthread_t pthread_self(void);
extern int   pthread_setcancelstate(int, int *);
extern int   pthread_setcanceltype(int, int *);
extern int   pthread_setconcurrency(int);
extern int   pthread_setschedparam(pthread_t, int ,
          const struct sched_param *);
extern int   pthread_setspecific(pthread_key_t, const void *);
extern void  pthread_testcancel(void);


