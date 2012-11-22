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

template <class Ptr>
struct pointer_traits
{
    typedef Ptr pointer;
};

template <class T>
struct pointer_traits<T*>
{
    typedef T* pointer;
    typedef T element_type;
    typedef ptrdiff_t difference_type;

};

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

}

}


