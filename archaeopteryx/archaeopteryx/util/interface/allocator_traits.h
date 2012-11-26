/*! \file   allocator_traits.h
	\date   Thursday November 15, 2012
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief  The header file for allocator_traits class.
*/

#pragma once

// Archaeopteryx Includes
#include <archaeopteryx/util/interface/functional.h>
#include <archaeopteryx/util/interface/iterator.h>
#include <archaeopteryx/util/interface/utility.h>

namespace archaeopteryx
{

namespace util
{

struct allocator_arg_t { };

allocator_arg_t allocator_arg = allocator_arg_t();

template <class T, class Alloc> struct uses_allocator;

// pointer_traits

template <class _Tp>
struct __has_element_type
{
private:
    struct __two {char __lx; char __lxx;};
    template <class _Up> static __two __test(...);
    template <class _Up> static char __test(typename _Up::element_type* = 0);
public:
    static const bool value = sizeof(__test<_Tp>(0)) == 1;
};

template <class _Ptr, bool = __has_element_type<_Ptr>::value>
struct __pointer_traits_element_type;

template <class _Ptr>
struct __pointer_traits_element_type<_Ptr, true>
{
    typedef typename _Ptr::element_type type;
};

template <template <class> class _Sp, class _Tp>
struct __pointer_traits_element_type<_Sp<_Tp>, true>
{
    typedef typename _Sp<_Tp>::element_type type;
};

template <template <class> class _Sp, class _Tp>
struct __pointer_traits_element_type<_Sp<_Tp>, false>
{
    typedef _Tp type;
};

template <template <class, class> class _Sp, class _Tp, class _A0>
struct __pointer_traits_element_type<_Sp<_Tp, _A0>, true>
{
    typedef typename _Sp<_Tp, _A0>::element_type type;
};

template <template <class, class> class _Sp, class _Tp, class _A0>
struct __pointer_traits_element_type<_Sp<_Tp, _A0>, false>
{
    typedef _Tp type;
};

template <template <class, class, class> class _Sp, class _Tp, class _A0, class _A1>
struct __pointer_traits_element_type<_Sp<_Tp, _A0, _A1>, true>
{
    typedef typename _Sp<_Tp, _A0, _A1>::element_type type;
};

template <template <class, class, class> class _Sp, class _Tp, class _A0, class _A1>
struct __pointer_traits_element_type<_Sp<_Tp, _A0, _A1>, false>
{
    typedef _Tp type;
};

template <template <class, class, class, class> class _Sp, class _Tp, class _A0,
                                                           class _A1, class _A2>
struct __pointer_traits_element_type<_Sp<_Tp, _A0, _A1, _A2>, true>
{
    typedef typename _Sp<_Tp, _A0, _A1, _A2>::element_type type;
};

template <template <class, class, class, class> class _Sp, class _Tp, class _A0,
                                                           class _A1, class _A2>
struct __pointer_traits_element_type<_Sp<_Tp, _A0, _A1, _A2>, false>
{
    typedef _Tp type;
};

template <class _Tp>
struct __has_difference_type
{
private:
    struct __two {char __lx; char __lxx;};
    template <class _Up> static __two __test(...);
    template <class _Up> static char __test(typename _Up::difference_type* = 0);
public:
    static const bool value = sizeof(__test<_Tp>(0)) == 1;
};

template <class _Ptr, bool = __has_difference_type<_Ptr>::value>
struct __pointer_traits_difference_type
{
    typedef ptrdiff_t type;
};

template <class _Ptr>
struct __pointer_traits_difference_type<_Ptr, true>
{
    typedef typename _Ptr::difference_type type;
};

template <class _Tp, class _Up>
struct __has_rebind
{
private:
    struct __two {char __lx; char __lxx;};
    template <class _Xp> static __two __test(...);
    template <class _Xp> static char __test(typename _Xp::template rebind<_Up>* = 0);
public:
    static const bool value = sizeof(__test<_Tp>(0)) == 1;
};

template <class _Tp, class _Up, bool = __has_rebind<_Tp, _Up>::value>
struct __pointer_traits_rebind
{
    typedef typename _Tp::template rebind<_Up>::other type;
};

template <template <class> class _Sp, class _Tp, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp>, _Up, true>
{
    typedef typename _Sp<_Tp>::template rebind<_Up>::other type;
};

template <template <class> class _Sp, class _Tp, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp>, _Up, false>
{
    typedef _Sp<_Up> type;
};

template <template <class, class> class _Sp, class _Tp, class _A0, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp, _A0>, _Up, true>
{
    typedef typename _Sp<_Tp, _A0>::template rebind<_Up>::other type;
};

template <template <class, class> class _Sp, class _Tp, class _A0, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp, _A0>, _Up, false>
{
    typedef _Sp<_Up, _A0> type;
};

template <template <class, class, class> class _Sp, class _Tp, class _A0,
                                         class _A1, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp, _A0, _A1>, _Up, true>
{
    typedef typename _Sp<_Tp, _A0, _A1>::template rebind<_Up>::other type;
};

template <template <class, class, class> class _Sp, class _Tp, class _A0,
                                         class _A1, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp, _A0, _A1>, _Up, false>
{
    typedef _Sp<_Up, _A0, _A1> type;
};

template <template <class, class, class, class> class _Sp, class _Tp, class _A0,
                                                class _A1, class _A2, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp, _A0, _A1, _A2>, _Up, true>
{
    typedef typename _Sp<_Tp, _A0, _A1, _A2>::template rebind<_Up>::other type;
};

template <template <class, class, class, class> class _Sp, class _Tp, class _A0,
                                                class _A1, class _A2, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp, _A0, _A1, _A2>, _Up, false>
{
    typedef _Sp<_Up, _A0, _A1, _A2> type;
};

template <class _Ptr>
struct pointer_traits
{
    typedef _Ptr                                                     pointer;
    typedef typename __pointer_traits_element_type<pointer>::type    element_type;
    typedef typename __pointer_traits_difference_type<pointer>::type difference_type;

    template <class _Up> struct rebind
        {typedef typename __pointer_traits_rebind<pointer, _Up>::type other;};

private:
    struct __nat {};
public:
    static pointer pointer_to(typename conditional<is_void<element_type>::value,
                                           __nat, element_type>::type& __r)
        {return pointer::pointer_to(__r);}
};

template <class _Tp>
struct pointer_traits<_Tp*>
{
    typedef _Tp*      pointer;
    typedef _Tp       element_type;
    typedef ptrdiff_t difference_type;

    template <class _Up> struct rebind {typedef _Up* other;};

private:
    struct __nat {};
public:
    static pointer pointer_to(typename conditional<is_void<element_type>::value,
                                      __nat, element_type>::type& __r)
        {return addressof(__r);}
};

// allocator traits

template <class Alloc>
struct allocator_traits
{
    typedef Alloc                        allocator_type;
    typedef typename allocator_type::value_type
                                         value_type;

    typedef value_type* pointer;
    
    typedef const value_type* const_pointer;
    typedef void*             void_pointer;
    typedef const void*       const_void_pointer;
    
    typedef typename pointer_traits<pointer>::difference_type
                                         difference_type;
    typedef typename make_unsigned<difference_type>::type
                                         size_type;
    
    template <class U> struct rebind_alloc {typedef typename Alloc::template rebind<U>::other other;};
    template <class U> struct rebind_traits {typedef allocator_traits<U> other;};

    static pointer allocate(allocator_type& a, size_type n);
    static pointer allocate(allocator_type& a, size_type n, const_void_pointer hint);

    static void deallocate(allocator_type& a, pointer p, size_type n);

    template <class T>
        static void construct(allocator_type& a, T* p);

    template <class T>
        static void destroy(allocator_type& a, T* p);

    static size_type max_size(const allocator_type& a);

    static allocator_type
        select_on_container_copy_construction(const allocator_type& a);
};

template <class T>
class allocator
{
public:
    typedef size_t                                size_type;
    typedef ptrdiff_t                             difference_type;
    typedef T*                                    pointer;
    typedef const T*                              const_pointer;
    typedef typename add_lvalue_reference<T>::type       reference;
    typedef typename add_lvalue_reference<const T>::type const_reference;
    typedef T                                     value_type;

    template <class U> struct rebind {typedef allocator<U> other;};

    allocator();
    allocator(const allocator&);
    template <class U> allocator(const allocator<U>&);
    ~allocator();
    pointer address(reference x) const;
    const_pointer address(const_reference x) const;
    pointer allocate(size_type, const void* hint = 0);
    void deallocate(pointer p, size_type n);
    size_type max_size() const;
    template<class U>
        void construct(U* p);
    template <class U>
        void destroy(U* p);
};

template <>
class allocator<void>
{
public:
    typedef void*                                 pointer;
    typedef const void*                           const_pointer;
    typedef void                                  value_type;

    template <class _Up> struct rebind {typedef allocator<_Up> other;};
};

template <class T, class U>
bool operator==(const allocator<T>&, const allocator<U>&);

template <class T, class U>
bool operator!=(const allocator<T>&, const allocator<U>&);

template <class OutputIterator, class T>
class raw_storage_iterator
    : public iterator<output_iterator_tag,
                      T,                               // purposefully not C++03
                      ptrdiff_t,                       // purposefully not C++03
                      T*,                              // purposefully not C++03
                      raw_storage_iterator<OutputIterator, T>&>           // purposefully not C++03
{
public:
    explicit raw_storage_iterator(OutputIterator x);
    raw_storage_iterator& operator*();
    raw_storage_iterator& operator=(const T& element);
    raw_storage_iterator& operator++();
    raw_storage_iterator  operator++(int);
};

template <class T> pair<T*,ptrdiff_t> get_temporary_buffer(ptrdiff_t n);
template <class T> void               return_temporary_buffer(T* p);

template <class T> T* addressof(T& r);

template <class InputIterator, class ForwardIterator>
ForwardIterator
uninitialized_copy(InputIterator first, InputIterator last, ForwardIterator result);

template <class InputIterator, class Size, class ForwardIterator>
ForwardIterator
uninitialized_copy_n(InputIterator first, Size n, ForwardIterator result);

template <class ForwardIterator, class T>
void uninitialized_fill(ForwardIterator first, ForwardIterator last, const T& x);

template <class ForwardIterator, class Size, class T>
ForwardIterator
uninitialized_fill_n(ForwardIterator first, Size n, const T& x);

// Unique Ptr

namespace __pointer_type_imp
{

namespace __has_pointer_type_imp
{
    template <class _Up> static char test(typename _Up::pointer* = 0);
}

template <class _Tp>
struct __has_pointer_type
    : public integral_constant<bool,
      sizeof(__has_pointer_type_imp::test<_Tp>(0)) == 1>
{
};

template <class _Tp, class _Dp, bool = __has_pointer_type<_Dp>::value>
struct __pointer_type
{
    typedef typename _Dp::pointer type;
};

template <class _Tp, class _Dp>
struct __pointer_type<_Tp, _Dp, false>
{
    typedef _Tp* type;
};

}  // __pointer_type_imp

template <class _Tp, class _Dp>
struct __pointer_type
{
    typedef typename __pointer_type_imp::__pointer_type<_Tp,
    	typename remove_reference<_Dp>::type>::type type;
};


template <class _Tp>
struct default_delete
{
    default_delete() {}
    template <class _Up>
        default_delete(const default_delete<_Up>&) {}
    void operator() (_Tp* __ptr) const
    {
        delete __ptr;
    }
};

template <class _Tp>
struct default_delete<_Tp[]>
{
public:
    default_delete() {}
    template <class _Up>
        default_delete(const default_delete<_Up[]>&) {}
    template <class _Up>
        void operator() (_Up* __ptr) const
        {
            delete [] __ptr;
        }
};

template <class _Tp, class _Dp = default_delete<_Tp> >
class unique_ptr
{
public:
    typedef _Tp element_type;
    typedef _Dp deleter_type;
    typedef typename __pointer_type<_Tp, deleter_type>::type pointer;
private:
    pair<pointer, deleter_type> __ptr_;

    struct __nat {int __for_bool_;};

    typedef       typename remove_reference<deleter_type>::type& _Dp_reference;
    typedef const typename remove_reference<deleter_type>::type& _Dp_const_reference;
public:
    unique_ptr()
        : __ptr_(pointer())
        {
        }
        
    explicit unique_ptr(pointer __p)
        : __ptr_(move(__p))
        {
        }

    template <class _Up, class _Ep>
    unique_ptr& operator=(unique_ptr<_Up, _Ep> __u)
    {
        reset(__u.release());
        __ptr_.second() = forward<deleter_type>(__u.get_deleter());
        return *this;
    }

    unique_ptr(pointer __p, deleter_type __d)
        : __ptr_(move(__p), move(__d)) {}

    ~unique_ptr() {reset();}

    typename add_lvalue_reference<_Tp>::type operator*() const
        {return *__ptr_.first();}
    pointer operator->() const {return __ptr_.first();}
    pointer get() const {return __ptr_.first();}
          _Dp_reference get_deleter()
        {return __ptr_.second();}
    _Dp_const_reference get_deleter() const
        {return __ptr_.second();}
    operator bool() const
        {return __ptr_.first() != nullptr;}

    pointer release()
    {
        pointer __t = __ptr_.first();
        __ptr_.first() = pointer();
        return __t;
    }

    void reset(pointer __p = pointer())
    {
        pointer __tmp = __ptr_.first();
        __ptr_.first() = __p;
        if (__tmp)
            __ptr_.second()(__tmp);
    }

    void swap(unique_ptr& __u)
        {__ptr_.swap(__u.__ptr_);}
};

template <class _Tp, class _Dp>
class unique_ptr<_Tp[], _Dp>
{
public:
    typedef _Tp element_type;
    typedef _Dp deleter_type;
    typedef typename __pointer_type<_Tp, deleter_type>::type pointer;
private:
    pair<pointer, deleter_type> __ptr_;

    struct __nat {int __for_bool_;};

    typedef       typename remove_reference<deleter_type>::type& _Dp_reference;
    typedef const typename remove_reference<deleter_type>::type& _Dp_const_reference;
public:
    unique_ptr()
        : __ptr_(pointer())
        {
            static_assert(!is_pointer<deleter_type>::value,
                "unique_ptr constructed with null function pointer deleter");
        }

    explicit unique_ptr(pointer __p)
        : __ptr_(__p)
        {
            static_assert(!is_pointer<deleter_type>::value,
                "unique_ptr constructed with null function pointer deleter");
        }

    unique_ptr(pointer __p, deleter_type __d)
        : __ptr_(__p, forward<deleter_type>(__d)) {}

    ~unique_ptr() {reset();}

    typename add_lvalue_reference<_Tp>::type operator[](size_t __i) const
        {return __ptr_.first()[__i];}
    pointer get() const {return __ptr_.first();}
          _Dp_reference get_deleter()
        {return __ptr_.second();}
    _Dp_const_reference get_deleter() const
        {return __ptr_.second();}
    operator bool() const
        {return __ptr_.first() != nullptr;}

    pointer release()
    {
        pointer __t = __ptr_.first();
        __ptr_.first() = pointer();
        return __t;
    }

    void reset(pointer __p = pointer())
    {
        pointer __tmp = __ptr_.first();
        __ptr_.first() = __p;
        if (__tmp)
            __ptr_.second()(__tmp);
    }

    void swap(unique_ptr& __u) {__ptr_.swap(__u.__ptr_);}
};

template <class _Tp, class _Dp>
inline void
swap(unique_ptr<_Tp, _Dp>& __x, unique_ptr<_Tp, _Dp>& __y) {__x.swap(__y);}

template <class _T1, class _D1, class _T2, class _D2>
inline bool
operator==(const unique_ptr<_T1, _D1>& __x, const unique_ptr<_T2, _D2>& __y) {return __x.get() == __y.get();}

template <class _T1, class _D1, class _T2, class _D2>
inline bool
operator!=(const unique_ptr<_T1, _D1>& __x, const unique_ptr<_T2, _D2>& __y) {return !(__x == __y);}

template <class _T1, class _D1, class _T2, class _D2>
inline bool
operator< (const unique_ptr<_T1, _D1>& __x, const unique_ptr<_T2, _D2>& __y)
{
    typedef typename unique_ptr<_T1, _D1>::pointer _P1;
    typedef typename unique_ptr<_T2, _D2>::pointer _P2;
    typedef typename common_type<_P1, _P2>::type _V;
    return less<_V>()(__x.get(), __y.get());
}

template <class _T1, class _D1, class _T2, class _D2>
inline bool
operator> (const unique_ptr<_T1, _D1>& __x, const unique_ptr<_T2, _D2>& __y) {return __y < __x;}

template <class _T1, class _D1, class _T2, class _D2>
inline bool
operator<=(const unique_ptr<_T1, _D1>& __x, const unique_ptr<_T2, _D2>& __y) {return !(__y < __x);}

template <class _T1, class _D1, class _T2, class _D2>
inline bool
operator>=(const unique_ptr<_T1, _D1>& __x, const unique_ptr<_T2, _D2>& __y) {return !(__x < __y);}

template <class _Tp, class _Dp>
inline unique_ptr<_Tp, _Dp>
move(unique_ptr<_Tp, _Dp>& __t)
{
    return unique_ptr<_Tp, _Dp>(__rv<unique_ptr<_Tp, _Dp> >(__t));
}

}

}


