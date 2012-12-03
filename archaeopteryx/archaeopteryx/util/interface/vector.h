/*	\file   vector.h
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   November 14, 2012
	\brief  The header file for the vector class
*/

#pragma once

// Archaeopteryx Includes
#include <archaeopteryx/util/interface/algorithm.h>
#include <archaeopteryx/util/interface/algorithm.h>
#include <archaeopteryx/util/interface/split_buffer.h>
#include <archaeopteryx/util/interface/limits.h>
#include <archaeopteryx/util/interface/functional.h>
#include <archaeopteryx/util/interface/allocator_traits.h>

namespace archaeopteryx
{

namespace util
{

template <bool>
class __vector_base_common
{
protected:
    __vector_base_common() {}
    void __throw_length_error() const;
    void __throw_out_of_range() const;
};

template <bool __b>
void
__vector_base_common<__b>::__throw_length_error() const
{
    //assert(!"vector length_error");
}

template <bool __b>
void
__vector_base_common<__b>::__throw_out_of_range() const
{
   // assert(!"vector out_of_range");
}

template <class _Tp, class _Allocator>
class __vector_base
    : protected __vector_base_common<true>
{
protected:
    typedef _Tp                                      value_type;
    typedef _Allocator                               allocator_type;
    typedef allocator_traits<allocator_type>         __alloc_traits;
    typedef value_type&                              reference;
    typedef const value_type&                        const_reference;
    typedef typename __alloc_traits::size_type       size_type;
    typedef typename __alloc_traits::difference_type difference_type;
    typedef typename __alloc_traits::pointer         pointer;
    typedef typename __alloc_traits::const_pointer   const_pointer;
    typedef pointer                                  iterator;
    typedef const_pointer                            const_iterator;

    pointer                                         __begin_;
    pointer                                         __end_;
    pair<pointer, allocator_type> __end_cap_;

    allocator_type& __alloc()
        {return __end_cap_.second;}
    const allocator_type& __alloc() const
        {return __end_cap_.second;}
    pointer& __end_cap()
        {return __end_cap_.first;}
    const pointer& __end_cap() const
        {return __end_cap_.first;}

    __vector_base();
    __vector_base(const allocator_type& __a);
    ~__vector_base();

    void clear() {__destruct_at_end(__begin_);}
    size_type capacity() const
        {return static_cast<size_type>(__end_cap() - __begin_);}

    void __destruct_at_end(const_pointer __new_last)
        {__destruct_at_end(__new_last, false_type());}
    void __destruct_at_end(const_pointer __new_last, false_type);
    void __destruct_at_end(const_pointer __new_last, true_type);

    void __copy_assign_alloc(const __vector_base& __c)
        {__copy_assign_alloc(__c, integral_constant<bool,
                      __alloc_traits::propagate_on_container_copy_assignment::value>());}

    void __move_assign_alloc(__vector_base& __c)
        {__move_assign_alloc(__c, integral_constant<bool,
                      __alloc_traits::propagate_on_container_move_assignment::value>());}

    static void __swap_alloc(allocator_type& __x, allocator_type& __y)
        {__swap_alloc(__x, __y, integral_constant<bool,
                      __alloc_traits::propagate_on_container_swap::value>());}
private:
    void __copy_assign_alloc(const __vector_base& __c, true_type)
        {
            if (__alloc() != __c.__alloc())
            {
                clear();
                __alloc_traits::deallocate(__alloc(), __begin_, capacity());
                __begin_ = __end_ = __end_cap() = nullptr;
            }
            __alloc() = __c.__alloc();
        }

    void __copy_assign_alloc(const __vector_base&, false_type)
        {}

    void __move_assign_alloc(__vector_base& __c, true_type)
        {
            __alloc() = _VSTD::move(__c.__alloc());
        }

    void __move_assign_alloc(__vector_base&, false_type)
       
        {}

    static void __swap_alloc(allocator_type& __x, allocator_type& __y, true_type)
        {
            util::swap(__x, __y);
        }
    static void __swap_alloc(allocator_type&, allocator_type&, false_type)
       
        {}
};

template <class _Tp, class _Allocator>
inline
void
__vector_base<_Tp, _Allocator>::__destruct_at_end(const_pointer __new_last, false_type)
{
    while (__new_last != __end_)
        __alloc_traits::destroy(__alloc(), const_cast<pointer>(--__end_));
}

template <class _Tp, class _Allocator>
inline
void
__vector_base<_Tp, _Allocator>::__destruct_at_end(const_pointer __new_last, true_type)
{
    __end_ = const_cast<pointer>(__new_last);
}

template <class _Tp, class _Allocator>
inline
__vector_base<_Tp, _Allocator>::__vector_base()
    : __begin_(0),
      __end_(0),
      __end_cap_(0, _Allocator())
{
}

template <class _Tp, class _Allocator>
inline
__vector_base<_Tp, _Allocator>::__vector_base(const allocator_type& __a)
    : __begin_(0),
      __end_(0),
      __end_cap_(0, __a)
{
}

template <class _Tp, class _Allocator>
__vector_base<_Tp, _Allocator>::~__vector_base()
{
    if (__begin_ != 0)
    {
        clear();
        __alloc_traits::deallocate(__alloc(), __begin_, capacity());
    }
}

template <class _Tp, class _Allocator = allocator<_Tp> >
class vector
    : private __vector_base<_Tp, _Allocator>
{
private:
    typedef __vector_base<_Tp, _Allocator>           __base;
public:
    typedef vector                                   __self;
    typedef _Tp                                      value_type;
    typedef _Allocator                               allocator_type;
    typedef typename __base::__alloc_traits          __alloc_traits;
    typedef typename __base::reference               reference;
    typedef typename __base::const_reference         const_reference;
    typedef typename __base::size_type               size_type;
    typedef typename __base::difference_type         difference_type;
    typedef typename __base::pointer                 pointer;
    typedef typename __base::const_pointer           const_pointer;
    typedef __wrap_iter<pointer>                     iterator;
    typedef __wrap_iter<const_pointer>               const_iterator;
    typedef util::reverse_iterator<iterator>         reverse_iterator;
    typedef util::reverse_iterator<const_iterator>   const_reverse_iterator;

    vector()
        {

        }
    explicit vector(const allocator_type& __a)
        : __base(__a)
    {

    }
    explicit vector(size_type __n);
    vector(size_type __n, const_reference __x);
    vector(size_type __n, const_reference __x, const allocator_type& __a);
    template <class _InputIterator>
        vector(_InputIterator __first, _InputIterator __last,
               typename enable_if<__is_input_iterator  <_InputIterator>::value &&
                                 !__is_forward_iterator<_InputIterator>::value>::type* = 0);
    template <class _InputIterator>
        vector(_InputIterator __first, _InputIterator __last, const allocator_type& __a,
               typename enable_if<__is_input_iterator  <_InputIterator>::value &&
                                 !__is_forward_iterator<_InputIterator>::value>::type* = 0);
    template <class _ForwardIterator>
        vector(_ForwardIterator __first, _ForwardIterator __last,
               typename enable_if<__is_forward_iterator<_ForwardIterator>::value>::type* = 0);
    template <class _ForwardIterator>
        vector(_ForwardIterator __first, _ForwardIterator __last, const allocator_type& __a,
               typename enable_if<__is_forward_iterator<_ForwardIterator>::value>::type* = 0);

    vector(const vector& __x);
    vector(const vector& __x, const allocator_type& __a);
    vector& operator=(const vector& __x);

    template <class _InputIterator>
        typename enable_if
        <
             __is_input_iterator  <_InputIterator>::value &&
            !__is_forward_iterator<_InputIterator>::value,
            void
        >::type
        assign(_InputIterator __first, _InputIterator __last);
    template <class _ForwardIterator>
        typename enable_if
        <
            __is_forward_iterator<_ForwardIterator>::value,
            void
        >::type
        assign(_ForwardIterator __first, _ForwardIterator __last);

    void assign(size_type __n, const_reference __u);

    allocator_type get_allocator() const
        {return this->__alloc();}

    iterator               begin();
    const_iterator         begin()   const;
    iterator               end();
    const_iterator         end()     const;

    reverse_iterator       rbegin()
        {return       reverse_iterator(end());}
    const_reverse_iterator rbegin()  const
        {return const_reverse_iterator(end());}
    reverse_iterator       rend()
        {return       reverse_iterator(begin());}
    const_reverse_iterator rend()    const
        {return const_reverse_iterator(begin());}

    const_iterator         cbegin()  const
        {return begin();}
    const_iterator         cend()    const
        {return end();}
    const_reverse_iterator crbegin() const
        {return rbegin();}
    const_reverse_iterator crend()   const
        {return rend();}

    size_type size() const
        {return static_cast<size_type>(this->__end_ - this->__begin_);}
    size_type capacity() const
        {return __base::capacity();}
    bool empty() const
        {return this->__begin_ == this->__end_;}
    size_type max_size() const;
    void reserve(size_type __n);
    void shrink_to_fit();

    reference       operator[](size_type __n);
    const_reference operator[](size_type __n) const;
    reference       at(size_type __n);
    const_reference at(size_type __n) const;

    reference       front()
    {
        return *this->__begin_;
    }
    const_reference front() const
    {
        return *this->__begin_;
    }
    reference       back()
    {
        return *(this->__end_ - 1);
    }

    const_reference back()  const
    {
        return *(this->__end_ - 1);
    }

    value_type*       data()
        {return _VSTD::__to_raw_pointer(this->__begin_);}
    const value_type* data() const
        {return _VSTD::__to_raw_pointer(this->__begin_);}

    void push_back(const_reference __x);
	void pop_back();

    iterator insert(const_iterator __position, size_type __n, const_reference __x);
    
    iterator insert(const_iterator __position, const_reference __x);
    
    template <class _InputIterator>
        typename enable_if
        <
             __is_input_iterator  <_InputIterator>::value &&
            !__is_forward_iterator<_InputIterator>::value,
            iterator
        >::type
        insert(const_iterator __position, _InputIterator __first, _InputIterator __last);
    template <class _ForwardIterator>
        typename enable_if
        <
            __is_forward_iterator<_ForwardIterator>::value,
            iterator
        >::type
        insert(const_iterator __position, _ForwardIterator __first, _ForwardIterator __last);

    iterator erase(const_iterator __position);
    iterator erase(const_iterator __first, const_iterator __last);

    void clear()
    {
        __base::clear();
        __invalidate_all_iterators();
    }

    void resize(size_type __sz);
    void resize(size_type __sz, const_reference __x);

    void swap(vector&);

    bool __invariants() const;

private:
    void __invalidate_all_iterators();
    void allocate(size_type __n);
    void deallocate();
    size_type __recommend(size_type __new_size) const;
    void __construct_at_end(size_type __n);
    void __construct_at_end(size_type __n, const_reference __x);
    template <class _ForwardIterator>
        typename enable_if
        <
            __is_forward_iterator<_ForwardIterator>::value,
            void
        >::type
        __construct_at_end(_ForwardIterator __first, _ForwardIterator __last);
    void __move_construct_at_end(pointer __first, pointer __last);
    void __append(size_type __n);
    void __append(size_type __n, const_reference __x);
    iterator       __make_iter(pointer __p);
    const_iterator __make_iter(const_pointer __p) const;
    void __swap_out_circular_buffer(__split_buffer<value_type, allocator_type&>& __v);
    pointer __swap_out_circular_buffer(__split_buffer<value_type, allocator_type&>& __v, pointer __p);
    void __move_range(pointer __from_s, pointer __from_e, pointer __to);
    void __move_assign(vector& __c, true_type);
    void __move_assign(vector& __c, false_type);
    void __destruct_at_end(const_pointer __new_last)
    {
        __base::__destruct_at_end(__new_last);
    }
    template <class _Up>
        void
        __push_back_slow_path(_Up& __x);
};

template <class _Tp, class _Allocator>
void
vector<_Tp, _Allocator>::__swap_out_circular_buffer(__split_buffer<value_type, allocator_type&>& __v)
{
    __alloc_traits::__construct_backward(this->__alloc(), this->__begin_, this->__end_, __v.__begin_);
    _VSTD::swap(this->__begin_, __v.__begin_);
    _VSTD::swap(this->__end_, __v.__end_);
    _VSTD::swap(this->__end_cap(), __v.__end_cap());
    __v.__first_ = __v.__begin_;
    __invalidate_all_iterators();
}

template <class _Tp, class _Allocator>
typename vector<_Tp, _Allocator>::pointer
vector<_Tp, _Allocator>::__swap_out_circular_buffer(__split_buffer<value_type, allocator_type&>& __v, pointer __p)
{
    pointer __r = __v.__begin_;
    __alloc_traits::__construct_backward(this->__alloc(), this->__begin_, __p, __v.__begin_);
    __alloc_traits::__construct_forward(this->__alloc(), __p, this->__end_, __v.__end_);
    _VSTD::swap(this->__begin_, __v.__begin_);
    _VSTD::swap(this->__end_, __v.__end_);
    _VSTD::swap(this->__end_cap(), __v.__end_cap());
    __v.__first_ = __v.__begin_;
    __invalidate_all_iterators();
    return __r;
}

//  Allocate space for __n objects
//  throws length_error if __n > max_size()
//  throws (probably bad_alloc) if memory run out
//  Precondition:  __begin_ == __end_ == __end_cap() == 0
//  Precondition:  __n > 0
//  Postcondition:  capacity() == __n
//  Postcondition:  size() == 0
template <class _Tp, class _Allocator>
void
vector<_Tp, _Allocator>::allocate(size_type __n)
{
    if (__n > max_size())
        this->__throw_length_error();
    this->__begin_ = this->__end_ = __alloc_traits::allocate(this->__alloc(), __n);
    this->__end_cap() = this->__begin_ + __n;
}

template <class _Tp, class _Allocator>
void
vector<_Tp, _Allocator>::deallocate()
{
    if (this->__begin_ != 0)
    {
        clear();
        __alloc_traits::deallocate(this->__alloc(), this->__begin_, capacity());
        this->__begin_ = this->__end_ = this->__end_cap() = 0;
    }
}

template <class _Tp, class _Allocator>
typename vector<_Tp, _Allocator>::size_type
vector<_Tp, _Allocator>::max_size() const
{
    return util::min<size_type>(__alloc_traits::max_size(this->__alloc()),
    	numeric_limits<size_type>::max() / 2);  // end() >= begin(), always
}

//  Precondition:  __new_size > capacity()
template <class _Tp, class _Allocator>
inline
typename vector<_Tp, _Allocator>::size_type
vector<_Tp, _Allocator>::__recommend(size_type __new_size) const
{
    const size_type __ms = max_size();
    if (__new_size > __ms)
        this->__throw_length_error();
    const size_type __cap = capacity();
    if (__cap >= __ms / 2)
        return __ms;
    return _VSTD::max<size_type>(2*__cap, __new_size);
}

//  Default constructs __n objects starting at __end_
//  throws if construction throws
//  Precondition:  __n > 0
//  Precondition:  size() + __n <= capacity()
//  Postcondition:  size() == size() + __n
template <class _Tp, class _Allocator>
void
vector<_Tp, _Allocator>::__construct_at_end(size_type __n)
{
    allocator_type& __a = this->__alloc();
    do
    {
        __alloc_traits::construct(__a, util::__to_raw_pointer(this->__end_));
        ++this->__end_;
        --__n;
    } while (__n > 0);
}

//  Copy constructs __n objects starting at __end_ from __x
//  throws if construction throws
//  Precondition:  __n > 0
//  Precondition:  size() + __n <= capacity()
//  Postcondition:  size() == old size() + __n
//  Postcondition:  [i] == __x for all i in [size() - __n, __n)
template <class _Tp, class _Allocator>
inline
void
vector<_Tp, _Allocator>::__construct_at_end(size_type __n, const_reference __x)
{
    allocator_type& __a = this->__alloc();
    do
    {
        __alloc_traits::construct(__a,
        	util::__to_raw_pointer(this->__end_), __x);
        ++this->__end_;
        --__n;
    } while (__n > 0);
}

template <class _Tp, class _Allocator>
template <class _ForwardIterator>
typename enable_if
<
    __is_forward_iterator<_ForwardIterator>::value,
    void
>::type
vector<_Tp, _Allocator>::__construct_at_end(_ForwardIterator __first, _ForwardIterator __last)
{
    allocator_type& __a = this->__alloc();
    for (; __first != __last; ++__first)
    {
        __alloc_traits::construct(__a, util::__to_raw_pointer(this->__end_), *__first);
        ++this->__end_;
    }
}

template <class _Tp, class _Allocator>
void
vector<_Tp, _Allocator>::__move_construct_at_end(pointer __first, pointer __last)
{
    allocator_type& __a = this->__alloc();
    for (; __first != __last; ++__first)
    {
        __alloc_traits::construct(__a, util::__to_raw_pointer(this->__end_),
                                  util::move(*__first));
        ++this->__end_;
    }
}

//  Default constructs __n objects starting at __end_
//  throws if construction throws
//  Postcondition:  size() == size() + __n
//  Exception safety: strong.
template <class _Tp, class _Allocator>
void
vector<_Tp, _Allocator>::__append(size_type __n)
{
    if (static_cast<size_type>(this->__end_cap() - this->__end_) >= __n)
        this->__construct_at_end(__n);
    else
    {
        allocator_type& __a = this->__alloc();
        __split_buffer<value_type, allocator_type&> __v(__recommend(size() + __n), size(), __a);
        __v.__construct_at_end(__n);
        __swap_out_circular_buffer(__v);
    }
}

//  Default constructs __n objects starting at __end_
//  throws if construction throws
//  Postcondition:  size() == size() + __n
//  Exception safety: strong.
template <class _Tp, class _Allocator>
void
vector<_Tp, _Allocator>::__append(size_type __n, const_reference __x)
{
    if (static_cast<size_type>(this->__end_cap() - this->__end_) >= __n)
        this->__construct_at_end(__n, __x);
    else
    {
        allocator_type& __a = this->__alloc();
        __split_buffer<value_type, allocator_type&> __v(__recommend(size() + __n), size(), __a);
        __v.__construct_at_end(__n, __x);
        __swap_out_circular_buffer(__v);
    }
}

template <class _Tp, class _Allocator>
vector<_Tp, _Allocator>::vector(size_type __n)
{
    if (__n > 0)
    {
        allocate(__n);
        __construct_at_end(__n);
    }
}

template <class _Tp, class _Allocator>
vector<_Tp, _Allocator>::vector(size_type __n, const_reference __x)
{
    if (__n > 0)
    {
        allocate(__n);
        __construct_at_end(__n, __x);
    }
}

template <class _Tp, class _Allocator>
vector<_Tp, _Allocator>::vector(size_type __n, const_reference __x, const allocator_type& __a)
    : __base(__a)
{
    if (__n > 0)
    {
        allocate(__n);
        __construct_at_end(__n, __x);
    }
}

template <class _Tp, class _Allocator>
template <class _InputIterator>
vector<_Tp, _Allocator>::vector(_InputIterator __first, _InputIterator __last,
       typename enable_if<__is_input_iterator  <_InputIterator>::value &&
                         !__is_forward_iterator<_InputIterator>::value>::type*)
{
    for (; __first != __last; ++__first)
        push_back(*__first);
}

template <class _Tp, class _Allocator>
template <class _InputIterator>
vector<_Tp, _Allocator>::vector(_InputIterator __first, _InputIterator __last, const allocator_type& __a,
       typename enable_if<__is_input_iterator  <_InputIterator>::value &&
                         !__is_forward_iterator<_InputIterator>::value>::type*)
    : __base(__a)
{
    for (; __first != __last; ++__first)
        push_back(*__first);
}

template <class _Tp, class _Allocator>
template <class _ForwardIterator>
vector<_Tp, _Allocator>::vector(_ForwardIterator __first, _ForwardIterator __last,
                                typename enable_if<__is_forward_iterator<_ForwardIterator>::value>::type*)
{
    size_type __n = static_cast<size_type>(_VSTD::distance(__first, __last));
    if (__n > 0)
    {
        allocate(__n);
        __construct_at_end(__first, __last);
    }
}

template <class _Tp, class _Allocator>
template <class _ForwardIterator>
vector<_Tp, _Allocator>::vector(_ForwardIterator __first, _ForwardIterator __last, const allocator_type& __a,
                                typename enable_if<__is_forward_iterator<_ForwardIterator>::value>::type*)
    : __base(__a)
{
    size_type __n = static_cast<size_type>(_VSTD::distance(__first, __last));
    if (__n > 0)
    {
        allocate(__n);
        __construct_at_end(__first, __last);
    }
}

template <class _Tp, class _Allocator>
vector<_Tp, _Allocator>::vector(const vector& __x)
    : __base(__alloc_traits::select_on_container_copy_construction(__x.__alloc()))
{
    size_type __n = __x.size();
    if (__n > 0)
    {
        allocate(__n);
        __construct_at_end(__x.__begin_, __x.__end_);
    }
}

template <class _Tp, class _Allocator>
vector<_Tp, _Allocator>::vector(const vector& __x, const allocator_type& __a)
    : __base(__a)
{
    size_type __n = __x.size();
    if (__n > 0)
    {
        allocate(__n);
        __construct_at_end(__x.__begin_, __x.__end_);
    }
}

template <class _Tp, class _Allocator>
inline
vector<_Tp, _Allocator>&
vector<_Tp, _Allocator>::operator=(const vector& __x)
{
    if (this != &__x)
    {
        __base::__copy_assign_alloc(__x);
        assign(__x.__begin_, __x.__end_);
    }
    return *this;
}

template <class _Tp, class _Allocator>
template <class _InputIterator>
typename enable_if
<
     __is_input_iterator  <_InputIterator>::value &&
    !__is_forward_iterator<_InputIterator>::value,
    void
>::type
vector<_Tp, _Allocator>::assign(_InputIterator __first, _InputIterator __last)
{
    clear();
    for (; __first != __last; ++__first)
        push_back(*__first);
}

template <class _Tp, class _Allocator>
template <class _ForwardIterator>
typename enable_if
<
    __is_forward_iterator<_ForwardIterator>::value,
    void
>::type
vector<_Tp, _Allocator>::assign(_ForwardIterator __first, _ForwardIterator __last)
{
    typename iterator_traits<_ForwardIterator>::difference_type __new_size = _VSTD::distance(__first, __last);
    if (static_cast<size_type>(__new_size) <= capacity())
    {
        _ForwardIterator __mid = __last;
        bool __growing = false;
        if (static_cast<size_type>(__new_size) > size())
        {
            __growing = true;
            __mid =  __first;
            _VSTD::advance(__mid, size());
        }
        pointer __m = _VSTD::copy(__first, __mid, this->__begin_);
        if (__growing)
            __construct_at_end(__mid, __last);
        else
            this->__destruct_at_end(__m);
    }
    else
    {
        deallocate();
        allocate(__recommend(static_cast<size_type>(__new_size)));
        __construct_at_end(__first, __last);
    }
}

template <class _Tp, class _Allocator>
void
vector<_Tp, _Allocator>::assign(size_type __n, const_reference __u)
{
    if (__n <= capacity())
    {
        size_type __s = size();
        _VSTD::fill_n(this->__begin_, _VSTD::min(__n, __s), __u);
        if (__n > __s)
            __construct_at_end(__n - __s, __u);
        else
            this->__destruct_at_end(this->__begin_ + __n);
    }
    else
    {
        deallocate();
        allocate(__recommend(static_cast<size_type>(__n)));
        __construct_at_end(__n, __u);
    }
}

template <class _Tp, class _Allocator>
inline
typename vector<_Tp, _Allocator>::iterator
vector<_Tp, _Allocator>::__make_iter(pointer __p)
{
    return iterator(__p);
}

template <class _Tp, class _Allocator>
inline
typename vector<_Tp, _Allocator>::const_iterator
vector<_Tp, _Allocator>::__make_iter(const_pointer __p) const
{
    return const_iterator(__p);
}

template <class _Tp, class _Allocator>
inline
typename vector<_Tp, _Allocator>::iterator
vector<_Tp, _Allocator>::begin()
{
    return __make_iter(this->__begin_);
}

template <class _Tp, class _Allocator>
inline
typename vector<_Tp, _Allocator>::const_iterator
vector<_Tp, _Allocator>::begin() const
{
    return __make_iter(this->__begin_);
}

template <class _Tp, class _Allocator>
inline
typename vector<_Tp, _Allocator>::iterator
vector<_Tp, _Allocator>::end()
{
    return __make_iter(this->__end_);
}

template <class _Tp, class _Allocator>
inline
typename vector<_Tp, _Allocator>::const_iterator
vector<_Tp, _Allocator>::end() const
{
    return __make_iter(this->__end_);
}

template <class _Tp, class _Allocator>
inline
typename vector<_Tp, _Allocator>::reference
vector<_Tp, _Allocator>::operator[](size_type __n)
{
    return this->__begin_[__n];
}

template <class _Tp, class _Allocator>
inline
typename vector<_Tp, _Allocator>::const_reference
vector<_Tp, _Allocator>::operator[](size_type __n) const
{
    return this->__begin_[__n];
}

template <class _Tp, class _Allocator>
typename vector<_Tp, _Allocator>::reference
vector<_Tp, _Allocator>::at(size_type __n)
{
    if (__n >= size())
        this->__throw_out_of_range();
    return this->__begin_[__n];
}

template <class _Tp, class _Allocator>
typename vector<_Tp, _Allocator>::const_reference
vector<_Tp, _Allocator>::at(size_type __n) const
{
    if (__n >= size())
        this->__throw_out_of_range();
    return this->__begin_[__n];
}

template <class _Tp, class _Allocator>
void
vector<_Tp, _Allocator>::reserve(size_type __n)
{
    if (__n > capacity())
    {
        allocator_type& __a = this->__alloc();
        __split_buffer<value_type, allocator_type&> __v(__n, size(), __a);
        __swap_out_circular_buffer(__v);
    }
}

template <class _Tp, class _Allocator>
void
vector<_Tp, _Allocator>::shrink_to_fit()
{
    if (capacity() > size())
    {
        allocator_type& __a = this->__alloc();
        __split_buffer<value_type, allocator_type&> __v(size(), size(), __a);
        __swap_out_circular_buffer(__v);
    }
}

template <class _Tp, class _Allocator>
template <class _Up>
void
vector<_Tp, _Allocator>::__push_back_slow_path(_Up& __x)
{
    allocator_type& __a = this->__alloc();
    __split_buffer<value_type, allocator_type&> __v(__recommend(size() + 1), size(), __a);
    __alloc_traits::construct(__a, _VSTD::__to_raw_pointer(__v.__end_++), _VSTD::forward<_Up>(__x));
    __swap_out_circular_buffer(__v);
}

template <class _Tp, class _Allocator>
inline
void
vector<_Tp, _Allocator>::push_back(const_reference __x)
{
    if (this->__end_ != this->__end_cap())
    {
        __alloc_traits::construct(this->__alloc(),
                                  _VSTD::__to_raw_pointer(this->__end_), __x);
        ++this->__end_;
    }
    else
        __push_back_slow_path(__x);
}

template <class _Tp, class _Allocator>
inline
void
vector<_Tp, _Allocator>::pop_back()
{
    this->__destruct_at_end(this->__end_ - 1);
}

template <class _Tp, class _Allocator>
inline
typename vector<_Tp, _Allocator>::iterator
vector<_Tp, _Allocator>::erase(const_iterator __position)
{
    pointer __p = const_cast<pointer>(&*__position);
    iterator __r = __make_iter(__p);
    this->__destruct_at_end(_VSTD::move(__p + 1, this->__end_, __p));
    return __r;
}

template <class _Tp, class _Allocator>
typename vector<_Tp, _Allocator>::iterator
vector<_Tp, _Allocator>::erase(const_iterator __first, const_iterator __last)
{
    pointer __p = this->__begin_ + (__first - begin());
    iterator __r = __make_iter(__p);
    this->__destruct_at_end(_VSTD::move(__p + (__last - __first), this->__end_, __p));
    return __r;
}

template <class _Tp, class _Allocator>
void
vector<_Tp, _Allocator>::__move_range(pointer __from_s, pointer __from_e, pointer __to)
{
    pointer __old_last = this->__end_;
    difference_type __n = __old_last - __to;
    for (pointer __i = __from_s + __n; __i < __from_e; ++__i, ++this->__end_)
        __alloc_traits::construct(this->__alloc(),
                                  _VSTD::__to_raw_pointer(this->__end_),
                                  _VSTD::move(*__i));
    _VSTD::move_backward(__from_s, __from_s + __n, __old_last);
}

template <class _Tp, class _Allocator>
typename vector<_Tp, _Allocator>::iterator
vector<_Tp, _Allocator>::insert(const_iterator __position, const_reference __x)
{
    pointer __p = this->__begin_ + (__position - begin());
    if (this->__end_ < this->__end_cap())
    {
        if (__p == this->__end_)
        {
            __alloc_traits::construct(this->__alloc(),
                                  _VSTD::__to_raw_pointer(this->__end_), __x);
            ++this->__end_;
        }
        else
        {
            __move_range(__p, this->__end_, __p + 1);
            const_pointer __xr = pointer_traits<const_pointer>::pointer_to(__x);
            if (__p <= __xr && __xr < this->__end_)
                ++__xr;
            *__p = *__xr;
        }
    }
    else
    {
        allocator_type& __a = this->__alloc();
        __split_buffer<value_type, allocator_type&> __v(__recommend(size() + 1),
        	__p - this->__begin_, __a);
        __v.push_back(__x);
        __p = __swap_out_circular_buffer(__v, __p);
    }
    return __make_iter(__p);
}

template <class _Tp, class _Allocator>
typename vector<_Tp, _Allocator>::iterator
vector<_Tp, _Allocator>::insert(const_iterator __position, size_type __n, const_reference __x)
{
    pointer __p = this->__begin_ + (__position - begin());
    if (__n > 0)
    {
        if (__n <= static_cast<size_type>(this->__end_cap() - this->__end_))
        {
            size_type __old_n = __n;
            pointer __old_last = this->__end_;
            if (__n > static_cast<size_type>(this->__end_ - __p))
            {
                size_type __cx = __n - (this->__end_ - __p);
                __construct_at_end(__cx, __x);
                __n -= __cx;
            }
            if (__n > 0)
            {
                __move_range(__p, __old_last, __p + __old_n);
                const_pointer __xr = pointer_traits<const_pointer>::pointer_to(__x);
                if (__p <= __xr && __xr < this->__end_)
                    __xr += __old_n;
                _VSTD::fill_n(__p, __n, *__xr);
            }
        }
        else
        {
            allocator_type& __a = this->__alloc();
            __split_buffer<value_type, allocator_type&> __v(__recommend(size() + __n), __p - this->__begin_, __a);
            __v.__construct_at_end(__n, __x);
            __p = __swap_out_circular_buffer(__v, __p);
        }
    }
    return __make_iter(__p);
}

template <class _Tp, class _Allocator>
template <class _InputIterator>
typename enable_if
<
     __is_input_iterator  <_InputIterator>::value &&
    !__is_forward_iterator<_InputIterator>::value,
    typename vector<_Tp, _Allocator>::iterator
>::type
vector<_Tp, _Allocator>::insert(const_iterator __position, _InputIterator __first, _InputIterator __last)
{
    difference_type __off = __position - begin();
    pointer __p = this->__begin_ + __off;
    allocator_type& __a = this->__alloc();
    pointer __old_last = this->__end_;
    for (; this->__end_ != this->__end_cap() && __first != __last; ++__first)
    {
        __alloc_traits::construct(__a, _VSTD::__to_raw_pointer(this->__end_),
                                  *__first);
        ++this->__end_;
    }
    __split_buffer<value_type, allocator_type&> __v(__a);
    if (__first != __last)
    {
        __v.__construct_at_end(__first, __last);
        difference_type __old_size = __old_last - this->__begin_;
        difference_type __old_p = __p - this->__begin_;
        reserve(__recommend(size() + __v.size()));
        __p = this->__begin_ + __old_p;
        __old_last = this->__begin_ + __old_size;
    }
    __p = _VSTD::rotate(__p, __old_last, this->__end_);
    insert(__make_iter(__p), make_move_iterator(__v.begin()),
                                    make_move_iterator(__v.end()));
    return begin() + __off;
}

template <class _Tp, class _Allocator>
template <class _ForwardIterator>
typename enable_if
<
    __is_forward_iterator<_ForwardIterator>::value,
    typename vector<_Tp, _Allocator>::iterator
>::type
vector<_Tp, _Allocator>::insert(const_iterator __position, _ForwardIterator __first, _ForwardIterator __last)
{
    pointer __p = this->__begin_ + (__position - begin());
    difference_type __n = _VSTD::distance(__first, __last);
    if (__n > 0)
    {
        if (__n <= this->__end_cap() - this->__end_)
        {
            size_type __old_n = __n;
            pointer __old_last = this->__end_;
            _ForwardIterator __m = __last;
            difference_type __dx = this->__end_ - __p;
            if (__n > __dx)
            {
                __m = __first;
                _VSTD::advance(__m, this->__end_ - __p);
                __construct_at_end(__m, __last);
                __n = __dx;
            }
            if (__n > 0)
            {
                __move_range(__p, __old_last, __p + __old_n);
                _VSTD::copy(__first, __m, __p);
            }
        }
        else
        {
            allocator_type& __a = this->__alloc();
            __split_buffer<value_type, allocator_type&> __v(__recommend(size() + __n), __p - this->__begin_, __a);
            __v.__construct_at_end(__first, __last);
            __p = __swap_out_circular_buffer(__v, __p);
        }
    }
    return __make_iter(__p);
}

template <class _Tp, class _Allocator>
void
vector<_Tp, _Allocator>::resize(size_type __sz)
{
    size_type __cs = size();
    if (__cs < __sz)
        this->__append(__sz - __cs);
    else if (__cs > __sz)
        this->__destruct_at_end(this->__begin_ + __sz);
}

template <class _Tp, class _Allocator>
void
vector<_Tp, _Allocator>::resize(size_type __sz, const_reference __x)
{
    size_type __cs = size();
    if (__cs < __sz)
        this->__append(__sz - __cs, __x);
    else if (__cs > __sz)
        this->__destruct_at_end(this->__begin_ + __sz);
}

template <class _Tp, class _Allocator>
void
vector<_Tp, _Allocator>::swap(vector& __x)
{
    _VSTD::swap(this->__begin_, __x.__begin_);
    _VSTD::swap(this->__end_, __x.__end_);
    _VSTD::swap(this->__end_cap(), __x.__end_cap());
    __base::__swap_alloc(this->__alloc(), __x.__alloc());
}

template <class _Tp, class _Allocator>
bool
vector<_Tp, _Allocator>::__invariants() const
{
    if (this->__begin_ == 0)
    {
        if (this->__end_ != 0 || this->__end_cap() != 0)
            return false;
    }
    else
    {
        if (this->__begin_ > this->__end_)
            return false;
        if (this->__begin_ == this->__end_cap())
            return false;
        if (this->__end_ > this->__end_cap())
            return false;
    }
    return true;
}

template <class _Tp, class _Allocator>
inline
void
vector<_Tp, _Allocator>::__invalidate_all_iterators()
{

}



template <class _Tp, class _Allocator>
inline
bool
operator==(const vector<_Tp, _Allocator>& __x, const vector<_Tp, _Allocator>& __y)
{
    const typename vector<_Tp, _Allocator>::size_type __sz = __x.size();
    return __sz == __y.size() && _VSTD::equal(__x.begin(), __x.end(), __y.begin());
}

template <class _Tp, class _Allocator>
inline
bool
operator!=(const vector<_Tp, _Allocator>& __x, const vector<_Tp, _Allocator>& __y)
{
    return !(__x == __y);
}

template <class _Tp, class _Allocator>
inline
bool
operator< (const vector<_Tp, _Allocator>& __x, const vector<_Tp, _Allocator>& __y)
{
    return _VSTD::lexicographical_compare(__x.begin(), __x.end(), __y.begin(), __y.end());
}

template <class _Tp, class _Allocator>
inline
bool
operator> (const vector<_Tp, _Allocator>& __x, const vector<_Tp, _Allocator>& __y)
{
    return __y < __x;
}

template <class _Tp, class _Allocator>
inline
bool
operator>=(const vector<_Tp, _Allocator>& __x, const vector<_Tp, _Allocator>& __y)
{
    return !(__x < __y);
}

template <class _Tp, class _Allocator>
inline
bool
operator<=(const vector<_Tp, _Allocator>& __x, const vector<_Tp, _Allocator>& __y)
{
    return !(__y < __x);
}

template <class _Tp, class _Allocator>
inline
void
swap(vector<_Tp, _Allocator>& __x, vector<_Tp, _Allocator>& __y)
{
    __x.swap(__y);
}

}

}

