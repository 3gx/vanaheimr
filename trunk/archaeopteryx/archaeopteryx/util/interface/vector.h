/*	\file   vector.h
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   November 14, 2012
	\brief  The header file for the vector class
*/

#pragma once

namespace archaeopteryx
{

namespace util
{

template <class T, class Allocator = allocator<T> >
class vector
{
public:
    // types
	typedef T                                        value_type;
    typedef Allocator                                allocator_type;
    typedef typename allocator_type::reference       reference;
    typedef typename allocator_type::const_reference const_reference;
    typedef typename allocator_type::size_type       size_type;
    typedef typename allocator_type::difference_type difference_type;
    typedef typename allocator_type::pointer         pointer;
    typedef typename allocator_type::const_pointer   const_pointer;
    typedef pointer                                  iterator;
    typedef const_pointer                            const_iterator;
    typedef util::reverse_iterator<iterator>         reverse_iterator;
    typedef util::reverse_iterator<const_iterator>   const_reverse_iterator;

public:
	// construction
	__device__ vector();
    __device__ explicit vector(const allocator_type&);
    __device__ explicit vector(size_type n);
    __device__ vector(size_type n, const value_type& value, const allocator_type& =
		allocator_type());
    template <class InputIterator>
        __device__ vector(InputIterator first, InputIterator last, const allocator_type& =
			allocator_type());
    
	__device__ vector(const vector& x);
    __device__ ~vector();
    
	__device__ vector& operator=(const vector& x);
    template <class InputIterator>
        __device__ void assign(InputIterator first, InputIterator last);
    __device__ void assign(size_type n, const value_type& u);
	
public:
	// iteration
    __device__ iterator               begin();
    __device__ const_iterator         begin()   const;
    __device__ iterator               end();
    __device__ const_iterator         end()     const;

    __device__ reverse_iterator       rbegin();
    __device__ const_reverse_iterator rbegin()  const;
    __device__ reverse_iterator       rend();
    __device__ const_reverse_iterator rend()    const;

public:
	// capacity
    __device__ size_type size() const;
    __device__ size_type max_size() const;
    __device__ size_type capacity() const;
    __device__ bool empty() const;

public:
	// resize
    __device__ void reserve(size_type n);
    __device__ void shrink_to_fit();
    __device__ void resize(size_type sz);
    __device__ void resize(size_type sz, const value_type& c);


public:
	// element access
    __device__ reference       operator[](size_type n);
    __device__ const_reference operator[](size_type n) const;
    __device__ reference       at(size_type n);
    __device__ const_reference at(size_type n) const;

    __device__ reference       front();
    __device__ const_reference front() const;
    __device__ reference       back();
    __device__ const_reference back() const;

    __device__ value_type*       data();
    __device__ const value_type* data() const;

public:
	// insertion/deletion
    __device__ void push_back(const value_type& x);
    __device__ void pop_back();

    __device__ iterator insert(const_iterator position, const value_type& x);
    __device__ iterator insert(const_iterator position, size_type n, const value_type& x);
    template <class InputIterator>
        __device__ iterator insert(const_iterator position, InputIterator first,
			InputIterator last);

    __device__ iterator erase(const_iterator position);
    __device__ iterator erase(const_iterator first, const_iterator last);

    __device__ void clear();

    __device__ void swap(vector&);

public:
	// misc
	__device__ allocator_type get_allocator() const;

private:
	pointer _begin;
	pointer _end;
	pointer _capacityEnd;

	allocator_type _allocator;

};

}

}

#include <archaeopteryx/util/implementation/vector.inl>

