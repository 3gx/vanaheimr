/*	\file   map.h
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   November 14, 2012
	\brief  The header file for the map class
*/

#pragma once

// Archaeopteryx Includes
#include <archaeopteryx/util/interface/functional.h>
#include <archaeopteryx/util/interface/allocator_traits.h>
#include <archaeopteryx/util/interface/RedBlackTree.h>

namespace archaeopteryx
{

namespace util
{

template <class _Key, class _Tp, class _Compare, bool = is_empty<_Compare>::value>
class __map_value_compare
    : private _Compare
{
    typedef pair<typename remove_const<_Key>::type, _Tp> _Pp;
    typedef pair<const _Key, _Tp> _CP;
public:
    __map_value_compare()
        : _Compare() {}
    __map_value_compare(_Compare c)
        : _Compare(c) {}
    const _Compare& key_comp() const {return *this;}
    bool operator()(const _CP& __x, const _CP& __y) const
        {return static_cast<const _Compare&>(*this)(__x.first, __y.first);}
    bool operator()(const _CP& __x, const _Pp& __y) const
        {return static_cast<const _Compare&>(*this)(__x.first, __y.first);}
    bool operator()(const _CP& __x, const _Key& __y) const
        {return static_cast<const _Compare&>(*this)(__x.first, __y);}
    bool operator()(const _Pp& __x, const _CP& __y) const
        {return static_cast<const _Compare&>(*this)(__x.first, __y.first);}
    bool operator()(const _Pp& __x, const _Pp& __y) const
        {return static_cast<const _Compare&>(*this)(__x.first, __y.first);}
    bool operator()(const _Pp& __x, const _Key& __y) const
        {return static_cast<const _Compare&>(*this)(__x.first, __y);}
    bool operator()(const _Key& __x, const _CP& __y) const
        {return static_cast<const _Compare&>(*this)(__x, __y.first);}
    bool operator()(const _Key& __x, const _Pp& __y) const
        {return static_cast<const _Compare&>(*this)(__x, __y.first);}
    bool operator()(const _Key& __x, const _Key& __y) const
        {return static_cast<const _Compare&>(*this)(__x, __y);}
};

template <class _Key, class _Tp, class _Compare>
class __map_value_compare<_Key, _Tp, _Compare, false>
{
    _Compare comp;

    typedef pair<typename remove_const<_Key>::type, _Tp> _Pp;
    typedef pair<const _Key, _Tp> _CP;

public:
    __map_value_compare()
        : comp() {}
    __map_value_compare(_Compare c)
        : comp(c) {}
    const _Compare& key_comp() const {return comp;}

    bool operator()(const _CP& __x, const _CP& __y) const
        {return comp(__x.first, __y.first);}
    bool operator()(const _CP& __x, const _Pp& __y) const
        {return comp(__x.first, __y.first);}
    bool operator()(const _CP& __x, const _Key& __y) const
        {return comp(__x.first, __y);}
    bool operator()(const _Pp& __x, const _CP& __y) const
        {return comp(__x.first, __y.first);}
    bool operator()(const _Pp& __x, const _Pp& __y) const
        {return comp(__x.first, __y.first);}
    bool operator()(const _Pp& __x, const _Key& __y) const
        {return comp(__x.first, __y);}
    bool operator()(const _Key& __x, const _CP& __y) const
        {return comp(__x, __y.first);}
    bool operator()(const _Key& __x, const _Pp& __y) const
        {return comp(__x, __y.first);}
    bool operator()(const _Key& __x, const _Key& __y) const
        {return comp(__x, __y);}
};

template <class _Allocator>
class __map_node_destructor
{
    typedef _Allocator                          allocator_type;
    typedef allocator_traits<allocator_type>    __alloc_traits;
    typedef typename __alloc_traits::value_type::value_type value_type;
public:
    typedef typename __alloc_traits::pointer    pointer;
private:
    typedef typename value_type::first_type     first_type;
    typedef typename value_type::second_type    second_type;

    allocator_type& __na_;

    __map_node_destructor& operator=(const __map_node_destructor&);

public:
    bool __first_constructed;
    bool __second_constructed;

    explicit __map_node_destructor(allocator_type& __na)
        : __na_(__na),
          __first_constructed(false),
          __second_constructed(false)
        {}

    void operator()(pointer __p)
    {
        if (__second_constructed)
            __alloc_traits::destroy(__na_, _Vaddressof(__p->__value_.second));
        if (__first_constructed)
            __alloc_traits::destroy(__na_, _Vaddressof(__p->__value_.first));
        if (__p)
            __alloc_traits::deallocate(__na_, __p, 1);
    }
};

template <class _Key, class _Tp, class _Compare, class _Allocator>
    class map;
template <class _Key, class _Tp, class _Compare, class _Allocator>
    class multimap;
template <class _TreeIterator> class __map_const_iterator;

template <class _TreeIterator>
class __map_iterator
{
    _TreeIterator __i_;

    typedef typename _TreeIterator::__pointer_traits             __pointer_traits;
    typedef const typename _TreeIterator::value_type::first_type __key_type;
    typedef typename _TreeIterator::value_type::second_type      __mapped_type;
public:
    typedef bidirectional_iterator_tag                           iterator_category;
    typedef pair<__key_type, __mapped_type>                      value_type;
    typedef typename _TreeIterator::difference_type              difference_type;
    typedef value_type&                                          reference;
    typedef typename __pointer_traits::template
            rebind<value_type>::other                      pointer;

    __map_iterator() {}

    __map_iterator(_TreeIterator __i) : __i_(__i) {}

    reference operator*() const {return *operator->();}
    pointer operator->() const {return (pointer)__i_.operator->();}

    __map_iterator& operator++() {++__i_; return *this;}
    __map_iterator operator++(int)
    {
        __map_iterator __t(*this);
        ++(*this);
        return __t;
    }

    __map_iterator& operator--() {--__i_; return *this;}
    __map_iterator operator--(int)
    {
        __map_iterator __t(*this);
        --(*this);
        return __t;
    }

    friend bool operator==(const __map_iterator& __x, const __map_iterator& __y)
        {return __x.__i_ == __y.__i_;}
    friend 
    bool operator!=(const __map_iterator& __x, const __map_iterator& __y)
        {return __x.__i_ != __y.__i_;}

    template <class, class, class, class> friend class map;
    template <class, class, class, class> friend class multimap;
    template <class> friend class __map_const_iterator;
};

template <class _TreeIterator>
class __map_const_iterator
{
    _TreeIterator __i_;

    typedef typename _TreeIterator::__pointer_traits             __pointer_traits;
    typedef const typename _TreeIterator::value_type::first_type __key_type;
    typedef typename _TreeIterator::value_type::second_type      __mapped_type;
public:
    typedef bidirectional_iterator_tag                           iterator_category;
    typedef pair<__key_type, __mapped_type>                      value_type;
    typedef typename _TreeIterator::difference_type              difference_type;
    typedef const value_type&                                    reference;
    typedef typename __pointer_traits::template
            rebind<const value_type>::other                      pointer;

    __map_const_iterator() {}

    __map_const_iterator(_TreeIterator __i) : __i_(__i) {}
    __map_const_iterator(
            __map_iterator<typename _TreeIterator::__non_const_iterator> __i)
               
                : __i_(__i.__i_) {}

    reference operator*() const {return *operator->();}
    pointer operator->() const {return (pointer)__i_.operator->();}

    __map_const_iterator& operator++() {++__i_; return *this;}
    __map_const_iterator operator++(int)
    {
        __map_const_iterator __t(*this);
        ++(*this);
        return __t;
    }

    __map_const_iterator& operator--() {--__i_; return *this;}
    __map_const_iterator operator--(int)
    {
        __map_const_iterator __t(*this);
        --(*this);
        return __t;
    }

    friend bool operator==(const __map_const_iterator& __x, const __map_const_iterator& __y)
        {return __x.__i_ == __y.__i_;}
    friend bool operator!=(const __map_const_iterator& __x, const __map_const_iterator& __y)
        {return __x.__i_ != __y.__i_;}

    template <class, class, class, class> friend class map;
    template <class, class, class, class> friend class multimap;
    template <class, class, class> friend class __tree_const_iterator;
};


template <class Key, class T, class Compare = less<Key>,
          class Allocator = allocator<pair<const Key, T> > >
class map
{
public:
    // types:
	typedef Key                               key_type;
	typedef T                                 mapped_type;
	typedef pair<const key_type, mapped_type> value_type;
	typedef Compare                           key_compare;
	typedef Allocator                         allocator_type;
	
public:
    class value_compare : public binary_function<value_type, value_type, bool>
    {
    protected:
		key_compare comp;

	protected:
		value_compare(key_compare c) : comp(c) {}
	
	public:
		bool operator()(const value_type& __x, const value_type& __y) const
			{return comp(__x.first, __y.first);}

	private:
		friend class map;
    };


public:
	typedef RedBlackTree<value_type, value_compare, allocator_type> __base;
	typedef typename __base::__node_traits                    __node_traits;
	typedef allocator_traits<allocator_type>                  __alloc_traits;

public:
    typedef typename __alloc_traits::pointer            pointer;
    typedef typename __alloc_traits::const_pointer      const_pointer;
    typedef typename __alloc_traits::size_type          size_type;
    typedef typename __alloc_traits::difference_type    difference_type;
    typedef __map_iterator<typename __base::iterator>   iterator;
    typedef __map_const_iterator<typename __base::const_iterator> const_iterator;
    typedef util::reverse_iterator<iterator>                  reverse_iterator;
    typedef util::reverse_iterator<const_iterator>            const_reverse_iterator;

public:
	explicit map(const key_compare& __comp = key_compare())
        : __tree_(__vc(__comp)) {}

    explicit map(const key_compare& __comp, const allocator_type& __a)
        : __tree_(__vc(__comp), __a) {}

    template <class _InputIterator>
        map(_InputIterator __f, _InputIterator __l,
            const key_compare& __comp = key_compare())
        : __tree_(__vc(__comp))
        {
            insert(__f, __l);
        }

    template <class _InputIterator>
        map(_InputIterator __f, _InputIterator __l,
            const key_compare& __comp, const allocator_type& __a)
        : __tree_(__vc(__comp), __a)
        {
            insert(__f, __l);
        }

    map(const map& __m)
        : __tree_(__m.__tree_)
        {
            insert(__m.begin(), __m.end());
        }

    map& operator=(const map& __m)
        {
            __tree_ = __m.__tree_;
            return *this;
        }

    explicit map(const allocator_type& __a)
        : __tree_(__a)
        {
        }

    map(const map& __m, const allocator_type& __a)
        : __tree_(__m.__tree_.value_comp(), __a)
        {
            insert(__m.begin(), __m.end());
        }

public:
	// Iteration
          iterator begin() {return __tree_.begin();}
    const_iterator begin() const {return __tree_.begin();}
          iterator end() {return __tree_.end();}
    const_iterator end() const {return __tree_.end();}

          reverse_iterator rbegin() {return reverse_iterator(end());}
    const_reverse_iterator rbegin() const
        {return const_reverse_iterator(end());}
          reverse_iterator rend()
            {return       reverse_iterator(begin());}
    const_reverse_iterator rend() const
        {return const_reverse_iterator(begin());}

public:
	// Size
    bool      empty() const {return __tree_.size() == 0;}
    size_type size() const {return __tree_.size();}
    size_type max_size() const {return __tree_.max_size();}

public:
	// Element Access
    mapped_type& operator[](const key_type& __k);

          mapped_type& at(const key_type& __k);
    const mapped_type& at(const key_type& __k) const;

    pair<iterator, bool>
        insert(const value_type& __v) {return __tree_.__insert_unique(__v);}

    iterator
        insert(const_iterator __p, const value_type& __v)
            {return __tree_.__insert_unique(__p.__i_, __v);}

    template <class _InputIterator>
            void insert(_InputIterator __f, _InputIterator __l)
        {
            for (const_iterator __e = cend(); __f != __l; ++__f)
                insert(__e.__i_, *__f);
        }

    iterator erase(const_iterator __p) {return __tree_.erase(__p.__i_);}
    size_type erase(const key_type& __k)
        {return __tree_.__erase_unique(__k);}
    iterator  erase(const_iterator __f, const_iterator __l)
        {return __tree_.erase(__f.__i_, __l.__i_);}
    void clear() {__tree_.clear();}

    void swap(map& __m)
        {__tree_.swap(__m.__tree_);}

    iterator find(const key_type& __k)             {return __tree_.find(__k);}
    const_iterator find(const key_type& __k) const {return __tree_.find(__k);}
    size_type      count(const key_type& __k) const
        {return __tree_.__count_unique(__k);}
    iterator lower_bound(const key_type& __k)
        {return __tree_.lower_bound(__k);}
    const_iterator lower_bound(const key_type& __k) const
        {return __tree_.lower_bound(__k);}
    iterator upper_bound(const key_type& __k)
        {return __tree_.upper_bound(__k);}
    const_iterator upper_bound(const key_type& __k) const
        {return __tree_.upper_bound(__k);}
    pair<iterator,iterator> equal_range(const key_type& __k)
        {return __tree_.__equal_range_unique(__k);}
    pair<const_iterator,const_iterator> equal_range(const key_type& __k) const
        {return __tree_.__equal_range_unique(__k);}

public:
	// Member access
    allocator_type get_allocator() const {return __tree_.__alloc();}
    key_compare    key_comp()      const {return __tree_.value_comp().key_comp();}
    value_compare  value_comp()    const {return value_compare(__tree_.value_comp().key_comp());}


private:
	__base __tree_;

};

}

}



