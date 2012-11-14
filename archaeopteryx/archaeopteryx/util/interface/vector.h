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
	vector();
    explicit vector(const allocator_type&);
    explicit vector(size_type n);
    vector(size_type n, const value_type& value, const allocator_type& =
		allocator_type());
    template <class InputIterator>
        vector(InputIterator first, InputIterator last, const allocator_type& =
			allocator_type());
    
	vector(const vector& x);
    ~vector();
    
	vector& operator=(const vector& x);
    vector& operator=(vector&& x);
    template <class InputIterator>
        void assign(InputIterator first, InputIterator last);
    void assign(size_type n, const value_type& u);
	
public:
	// iteration
    iterator               begin();
    const_iterator         begin()   const;
    iterator               end();
    const_iterator         end()     const;

    reverse_iterator       rbegin();
    const_reverse_iterator rbegin()  const;
    reverse_iterator       rend();
    const_reverse_iterator rend()    const;

public:
	// capacity
    size_type size() const;
    size_type max_size() const;
    size_type capacity() const;
    bool empty() const;

public:
	// resize
    void reserve(size_type n);
    void shrink_to_fit();
    void resize(size_type sz);
    void resize(size_type sz, const value_type& c);


public:
	// element access
    reference       operator[](size_type n);
    const_reference operator[](size_type n) const;
    reference       at(size_type n);
    const_reference at(size_type n) const;

    reference       front();
    const_reference front() const;
    reference       back();
    const_reference back() const;

    value_type*       data();
    const value_type* data() const;

public:
	// insertion/deletion
    void push_back(const value_type& x);
    void pop_back();

    iterator insert(const_iterator position, const value_type& x);
    iterator insert(const_iterator position, size_type n, const value_type& x);
    template <class InputIterator>
        iterator insert(const_iterator position, InputIterator first,
			InputIterator last);

    iterator erase(const_iterator position);
    iterator erase(const_iterator first, const_iterator last);

    void clear();

    void swap(vector&);

public:
	// misc
	allocator_type get_allocator() const;

private:
	pointer _begin;
	pointer _end;
	pointer _capacityEnd;

	allocator_type _allocator;

};

}

}

#include <archaeopteryx/util/implementation/vector.inl>

