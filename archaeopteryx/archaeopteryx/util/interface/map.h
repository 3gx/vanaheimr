/*	\file   map.h
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   November 14, 2012
	\brief  The header file for the map class
*/

#pragma once

namespace archaeopteryx
{

namespace util
{

template <class Key, class T, class Compare = less<Key>,
          class Allocator = allocator<pair<const Key, T>>>
class map
{
public:
	// forward
	class Iterator;
	class ConstIterator;

public:
    // types:
	typedef Key                               key_type;
	typedef T                                 mapped_type;
	typedef pair<const key_type, mapped_type> value_type;
	typedef Compare                           key_compare;
	typedef Allocator                         allocator_type;
	
	typedef typename allocator_type::reference       reference;
	typedef typename allocator_type::const_reference const_reference;
	typedef typename allocator_type::pointer         pointer;
	typedef typename allocator_type::const_pointer   const_pointer;
	typedef typename allocator_type::size_type       size_type;
	typedef typename allocator_type::difference_type difference_type;

	typedef Iterator                              iterator;
	typedef ConstIterator                         const_iterator;
	typedef reverse_iterator<iterator>       reverse_iterator;
	typedef reverse_iterator<const_iterator> const_reverse_iterator;

public:
	



};

}

}



