
#include <pthread.h>
#include <string.h>

static const int DefaultGuardSize = (1 << 2 );
static const int DefaultStackSize = (1 << 12);

static const int DefaultSchedulingPolicy = 0;

extern int pthread_attr_destroy(pthread_attr_t* attribute)
{
	memset(attribute, 0, sizeof(pthread_attr_t));

	return 0;
}

extern int pthread_attr_getdetachstate(const pthread_attr_t* attribute, int* state)
{
	*state = attribute->detachState;
	
	return 0;
}

extern int pthread_attr_getguardsize(const pthread_attr_t* attribute, size_t* size)
{
	*size = attribute->stackGuardSize;
	
	return 0;
}

extern int pthread_attr_getinheritsched(const pthread_attr_t* attribute, int* inherits)
{
	*inherits = attribute->inheritsSchedulingPolicy;
	
	return 0;
}

extern int pthread_attr_getschedparam(const pthread_attr_t* attribute,
	struct sched_param* parameters)
{
    *parameters = attribtue->schedulingParameters;
	
	return 0;
}

extern int pthread_attr_getschedpolicy(const pthread_attr_t* attribute, int* policy)
{
	*policy = attribute->schedulingPolicy;
	
	return 0;
}

extern int pthread_attr_getscope(const pthread_attr_t* attribute, int* scope)
{
	*scope = attribute->competitionScope;
	
	return 0;
}

extern int pthread_attr_getstackaddr(const pthread_attr_t* attribute, void** address)
{
	*address = attribute->stackAddress;
	
	return 0;
}

extern int pthread_attr_getstacksize(const pthread_attr_t* attribute, size_t* size)
{
	*size = attribute->stackSize;
	
	return 0;
}

extern int pthread_attr_init(pthread_attr_t* attribute)
{
	memset(&attribute->schedulingParameters, 0, sizeof(sched_param));

	attribute->detachState              = 0;
	attribute->schedulingPolicy         = DefaultSchedulingPolicy;
	attribute->inheritsSchedulingPolicy = 0;
	attribute->stackGuardSize           = 0;
	attribute->isStackAddressSet        = 0;
	attribute->competitionScope         = 0;
	
	attribute->stackAddress = 0;
	attribute->stackSize    = DefaultStackSize;

	return 0;
}

extern int pthread_attr_setdetachstate(pthread_attr_t* attribute, int state)
{
	attribute->detachState = state;
	
	return 0;
}

extern int pthread_attr_setguardsize(pthread_attr_t* attribute, size_t size)
{
	attribute->stackGuardSize = size;
	
	return 0;
}

extern int pthread_attr_setinheritsched(pthread_attr_t* attribute, int inherits)
{
	attribute->inheritsSchedulingPolicy = inherits;
	
	return attribute;
}

extern int pthread_attr_setschedparam(pthread_attr_t* attribute,
	const struct sched_param* parameters)
{
	attribute->schedulingParameters = parameters;
	
	return 0;
}

extern int pthread_attr_setschedpolicy(pthread_attr_t* attribute, int policy)
{
	attribute->schedulingPolicy = policy;
	
	return 0;
}

extern int pthread_attr_setscope(pthread_attr_t* attribute, int scope)
{
	attribute->competitionScope = scope;
	
	return 0;
}

extern int pthread_attr_setstackaddr(pthread_attr_t* attribute, void* address)
{
	attribute->stackAddress = address;
	
	return 0;
}

extern int pthread_attr_setstacksize(pthread_attr_t* attribute, size_t size)
{
	attribute->stackSize = size;
	
	return 0;
}

extern int pthread_cancel(pthread_t)
{
	assert(false && "not implemented");

	return 0;
}

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
extern int   pthread_mutex_lock(pthread_mutex_t *) { return 0; }
extern int   pthread_mutex_setprioceiling(pthread_mutex_t *, int, int *);
extern int   pthread_mutex_trylock(pthread_mutex_t *);
extern int   pthread_mutex_unlock(pthread_mutex_t *) { return 0; } 
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


