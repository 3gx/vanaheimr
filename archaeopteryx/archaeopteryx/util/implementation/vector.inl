/*	\file   vector.inl
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   November 14, 2012
	\brief  The header file for the vector class
*/

// Archaeopteryx Includes
#include <archaeopteryx/util/interface/vector.h>

namespace archaeopteryx
{

namespace util
{

template <class T, class A>
vector<T, A>::vector()
: _begin(0), _end(0), _endCapacity(0)
{

}

template <class T, class A>
vector<T, A>::vector(const allocator_type& a)
: _begin(0), _end(0), _endCapacity(0), _allocator(a)
{

}

template <class T, class A>
vector<T, A>::vector(size_type n)
{
	if(n > 0)
	{
		allocate(n);
		constructAtEnd(n);
	}
}

template <class T, class A>
vector<T, A>::vector(size_type n, const value_type& value, const allocator_type& a)
{
	if(n > 0)
	{
		allocate(n);
		constructAtEnd(n, value);
	}
}

template <class T, class A>
template <class InputIterator>
	vector<T, A>::vector(InputIterator first, InputIterator last,
		const allocator_type& allocator)
{

}

template <class T, class A>    
vector<T, A>::vector(const vector& x);

template <class T, class A>    
vector<T, A>::~vector();
    
template <class T, class A>    
vector<T, A>& vector<T, A>::operator=(const vector& x);
template <class T, class A>    
vector<T, A>& vector<T, A>::operator=(vector&& x);

template <class T, class A>    
template <class InputIterator>
void vector<T, A>::assign(InputIterator first, InputIterator last);


template <class T, class A>
void vector<T, A>::assign(size_type n, const value_type& u);
	
template <class T, class A>
    vector<T, A>::iterator               vector<T, A>::begin();
template <class T, class A>
    vector<T, A>::const_iterator         vector<T, A>::begin()   const;
template <class T, class A>
    vector<T, A>::iterator               vector<T, A>::end();
template <class T, class A>
    vector<T, A>::const_iterator         vector<T, A>::end()     const;

template <class T, class A>
    vector<T, A>::reverse_iterator       vector<T, A>::rbegin();
template <class T, class A>
    vector<T, A>::const_reverse_iterator vector<T, A>::rbegin()  const;
template <class T, class A>
    vector<T, A>::reverse_iterator       vector<T, A>::rend();
template <class T, class A>
    vector<T, A>::const_reverse_iterator vector<T, A>::rend()    const;

template <class T, class A>
    vector<T, A>::size_type vector<T, A>::size() const
{
	return _end - _begin;
}

template <class T, class A>
    vector<T, A>::size_type vector<T, A>::max_size() const;

template <class T, class A>
    vector<T, A>::size_type vector<T, A>::capacity() const
{
	return _capacityEnd - _begin;
}

template <class T, class A>
    bool vector<T, A>::empty() const
{
	return size() == 0;
}

template <class T, class A>
    void vector<T, A>::reserve(size_type n);
template <class T, class A>
    void vector<T, A>::shrink_to_fit();
template <class T, class A>
    void vector<T, A>::resize(size_type sz);
template <class T, class A>
    void vector<T, A>::resize(size_type sz, const value_type& c);


template <class T, class A>
    vector<T, A>::reference       vector<T, A>::operator[](size_type n);
template <class T, class A>
    vector<T, A>::const_reference vector<T, A>::operator[](size_type n) const;
template <class T, class A>
    vector<T, A>::reference       vector<T, A>::at(size_type n);
template <class T, class A>
    vector<T, A>::const_reference vector<T, A>::at(size_type n) const;

template <class T, class A>
    vector<T, A>::reference       vector<T, A>::front();
template <class T, class A>
    vector<T, A>::const_reference vector<T, A>::front() const;
template <class T, class A>
    vector<T, A>::reference       vector<T, A>::back();
template <class T, class A>
    vector<T, A>::const_reference vector<T, A>::back() const;

template <class T, class A>
    vector<T, A>::value_type*       vector<T, A>::data();
template <class T, class A>
    const vector<T, A>::value_type* vector<T, A>::data() const;

template <class T, class A>
    void vector<T, A>::push_back(const value_type& x);
template <class T, class A>
    void vector<T, A>::pop_back();

template <class T, class A>
    vector<T, A>::iterator vector<T, A>::insert(const_iterator position, const value_type& x);
template <class T, class A>
    vector<T, A>::iterator vector<T, A>::insert(const_iterator position, size_type n, const value_type& x);
template <class T, class A>
    template <class InputIterator>
        vector<T, A>::iterator vector<T, A>::insert(const_iterator position, InputIterator first,
			InputIterator last);

template <class T, class A>
    vector<T, A>::iterator vector<T, A>::erase(const_iterator position);
template <class T, class A>
    vector<T, A>::iterator vector<T, A>::erase(const_iterator first, const_iterator last);

template <class T, class A>
    void vector<T, A>::clear();

template <class T, class A>
    void vector<T, A>::swap(vector&);

template <class T, class A>
	vector<T, A>::allocator_type vector<T, A>::get_allocator() const;

}

}


