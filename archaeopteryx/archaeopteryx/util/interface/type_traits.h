/*	\file   type_traits.h
	\author Gregory Diamos <solusstultus@gmail.com>
	\date   November 14, 2012
	\brief  The header file for stl type traits.
*/

#pragma once

namespace archaeopteryx
{

namespace util
{

// helper class:
template <class T, T v> struct integral_constant;
typedef integral_constant<bool, true>  true_type;
typedef integral_constant<bool, false> false_type;

// helper traits
template <bool, class T = void> struct enable_if;
template <bool, class T, class F> struct conditional;

// Reference transformations:
template <class T> struct remove_reference;
template <class T> struct add_lvalue_reference;
template <class T> struct add_rvalue_reference;

// Primary classification traits:
template <class T> struct is_void;
template <class T> struct is_integral;
template <class T> struct is_floating_point;
template <class T> struct is_array;
template <class T> struct is_pointer;
template <class T> struct is_lvalue_reference;
template <class T> struct is_rvalue_reference;
template <class T> struct is_member_object_pointer;
template <class T> struct is_member_function_pointer;
template <class T> struct is_enum;
template <class T> struct is_union;
template <class T> struct is_class;
template <class T> struct is_function;

// Integral properties:
template <class T> struct is_signed;
template <class T> struct is_unsigned;
template <class T> struct make_signed;
template <class T> struct make_unsigned;

// Relationships between types:
template <class T, class U> struct is_same;
template <class Base, class Derived> struct is_base_of;
template <class From, class To> struct is_convertible;

// Const-volatile properties and transformations:
template <class T> struct is_const;
template <class T> struct is_volatile;
template <class T> struct remove_const;
template <class T> struct remove_volatile;
template <class T> struct remove_cv;
template <class T> struct add_const;
template <class T> struct add_volatile;
template <class T> struct add_cv;

// Member introspection:
template <class T> struct is_pod;
template <class T> struct is_trivial;
template <class T> struct is_trivially_copyable;
template <class T> struct is_standard_layout;
template <class T> struct is_literal_type;
template <class T> struct is_empty;
template <class T> struct is_polymorphic;
template <class T> struct is_abstract;

template <class T> struct is_constructible;
template <class T>                struct is_default_constructible;
template <class T>                struct is_copy_constructible;
template <class T>                struct is_move_constructible;
template <class T, class U>       struct is_assignable;
template <class T>                struct is_copy_assignable;
template <class T>                struct is_move_assignable;
template <class T>                struct is_destructible;

template <class T> struct is_trivially_constructible;
template <class T>                struct is_trivially_default_constructible;
template <class T>                struct is_trivially_copy_constructible;
template <class T>                struct is_trivially_move_constructible;
template <class T>       struct is_trivially_assignable;
template <class T>                struct is_trivially_copy_assignable;
template <class T>                struct is_trivially_move_assignable;
template <class T>                struct is_trivially_destructible;

// helper class:

template <class _Tp, _Tp __v>
struct integral_constant
{
    static const _Tp value = __v;
    typedef _Tp value_type;
    typedef integral_constant type;
    operator value_type() const {return value;}
};

template <class _Tp, _Tp __v>
const _Tp integral_constant<_Tp, __v>::value;

typedef integral_constant<bool, true>  true_type;
typedef integral_constant<bool, false> false_type;


// helper traits
template <bool _Bp, class _If, class _Then>
    struct conditional {typedef _If type;};
template <class _If, class _Then>
    struct conditional<false, _If, _Then> {typedef _Then type;};

template <bool, class _Tp> struct enable_if {};
template <class _Tp> struct enable_if<true, _Tp> {typedef _Tp type;};


// is const

template <class _Tp> struct is_const            : public false_type {};
template <class _Tp> struct is_const<_Tp const> : public true_type {};

// is_volatile

template <class _Tp> struct is_volatile               : public false_type {};
template <class _Tp> struct is_volatile<_Tp volatile> : public true_type {};

// remove_const

template <class _Tp> struct remove_const            {typedef _Tp type;};
template <class _Tp> struct remove_const<const _Tp> {typedef _Tp type;};

// remove_volatile

template <class _Tp> struct remove_volatile               {typedef _Tp type;};
template <class _Tp> struct remove_volatile<volatile _Tp> {typedef _Tp type;};

// remove_cv

template <class _Tp> struct remove_cv
{typedef typename remove_volatile<typename remove_const<_Tp>::type>::type type;};

// is_void

template <class _Tp> struct __is_void       : public false_type {};
template <>          struct __is_void<void> : public true_type {};

template <class _Tp> struct is_void
    : public __is_void<typename remove_cv<_Tp>::type> {};

// __is_nullptr_t

template <class _Tp> struct ____is_nullptr_t       : public false_type {};

// is_integral

template <class _Tp> struct __is_integral                     : public false_type {};
template <>          struct __is_integral<bool>               : public true_type {};
template <>          struct __is_integral<char>               : public true_type {};
template <>          struct __is_integral<signed char>        : public true_type {};
template <>          struct __is_integral<unsigned char>      : public true_type {};
template <>          struct __is_integral<wchar_t>            : public true_type {};
template <>          struct __is_integral<short>              : public true_type {};
template <>          struct __is_integral<unsigned short>     : public true_type {};
template <>          struct __is_integral<int>                : public true_type {};
template <>          struct __is_integral<unsigned int>       : public true_type {};
template <>          struct __is_integral<long>               : public true_type {};
template <>          struct __is_integral<unsigned long>      : public true_type {};
template <>          struct __is_integral<long long>          : public true_type {};
template <>          struct __is_integral<unsigned long long> : public true_type {};

template <class _Tp> struct is_integral
    : public __is_integral<typename remove_cv<_Tp>::type> {};

// is_floating_point

template <class _Tp> struct __is_floating_point              : public false_type {};
template <>          struct __is_floating_point<float>       : public true_type {};
template <>          struct __is_floating_point<double>      : public true_type {};
template <>          struct __is_floating_point<long double> : public true_type {};

template <class _Tp> struct is_floating_point
    : public __is_floating_point<typename remove_cv<_Tp>::type> {};

// is_array

template <class _Tp> struct is_array
    : public false_type {};
template <class _Tp> struct is_array<_Tp[]>
    : public true_type {};
template <class _Tp, size_t _Np> struct is_array<_Tp[_Np]>
    : public true_type {};

// is_pointer

template <class _Tp> struct __is_pointer       : public false_type {};
template <class _Tp> struct __is_pointer<_Tp*> : public true_type {};

template <class _Tp> struct is_pointer
    : public __is_pointer<typename remove_cv<_Tp>::type> {};

// is_reference

template <class _Tp> struct is_lvalue_reference       : public false_type {};
template <class _Tp> struct is_lvalue_reference<_Tp&> : public true_type {};

template <class _Tp> struct is_rvalue_reference        : public false_type {};

template <class _Tp> struct is_reference        : public false_type {};
template <class _Tp> struct is_reference<_Tp&>  : public true_type {};

// is_union

template <class _Tp> struct __libcpp_union : public false_type {};
template <class _Tp> struct is_union
    : public __libcpp_union<typename remove_cv<_Tp>::type> {};


// is_class

struct __two {char __lx[2];};

namespace __is_class_imp
{
template <class _Tp> char  __test(int _Tp::*);
template <class _Tp> __two __test(...);
}

template <class _Tp> struct is_class
    : public integral_constant<bool, sizeof(__is_class_imp::__test<_Tp>(0)) == 1 && !is_union<_Tp>::value> {};


// is_same

template <class _Tp, class _Up> struct is_same           : public false_type {};
template <class _Tp>            struct is_same<_Tp, _Tp> : public true_type {};

// is_function

namespace __is_function_imp
{
template <class _Tp> char  __test(_Tp*);
template <class _Tp> __two __test(...);
template <class _Tp> _Tp&  __source();
}

template <class _Tp, bool = is_class<_Tp>::value ||
                            is_union<_Tp>::value ||
                            is_void<_Tp>::value  ||
                            is_reference<_Tp>::value >
struct __is_function
    : public integral_constant<bool, sizeof(__is_function_imp::__test<_Tp>(__is_function_imp::__source<_Tp>())) == 1>
    {};
template <class _Tp> struct __is_function<_Tp, true> : public false_type {};

template <class _Tp> struct is_function
    : public __is_function<_Tp> {};

// is_member_function_pointer

template <class _Tp> struct            __is_member_function_pointer             : public false_type {};
template <class _Tp, class _Up> struct __is_member_function_pointer<_Tp _Up::*> : public is_function<_Tp> {};

template <class _Tp> struct is_member_function_pointer
    : public __is_member_function_pointer<typename remove_cv<_Tp>::type> {};

// is_member_pointer

template <class _Tp>            struct __is_member_pointer             : public false_type {};
template <class _Tp, class _Up> struct __is_member_pointer<_Tp _Up::*> : public true_type {};

template <class _Tp> struct is_member_pointer
    : public __is_member_pointer<typename remove_cv<_Tp>::type> {};

// is_member_object_pointer

template <class _Tp> struct is_member_object_pointer
    : public integral_constant<bool, is_member_pointer<_Tp>::value &&
                                    !is_member_function_pointer<_Tp>::value> {};

// is_enum
template <class _Tp> struct is_enum
    : public integral_constant<bool, !is_void<_Tp>::value             &&
                                     !is_integral<_Tp>::value         &&
                                     !is_floating_point<_Tp>::value   &&
                                     !is_array<_Tp>::value            &&
                                     !is_pointer<_Tp>::value          &&
                                     !is_reference<_Tp>::value        &&
                                     !is_member_pointer<_Tp>::value   &&
                                     !is_union<_Tp>::value            &&
                                     !is_class<_Tp>::value            &&
                                     !is_function<_Tp>::value         > {};

// remove_reference

template <class _Tp> struct remove_reference        {typedef _Tp type;};
template <class _Tp> struct remove_reference<_Tp&>  {typedef _Tp type;};

// add_lvalue_reference

template <class _Tp> struct add_lvalue_reference {typedef _Tp& type;};
//template <class _Tp> struct add_lvalue_reference<_Tp&> {typedef _Tp& type;};  // for older compiler
template <>          struct add_lvalue_reference<void> {typedef void type;};
template <>          struct add_lvalue_reference<const void> {typedef const void type;};
template <>          struct add_lvalue_reference<volatile void> {typedef volatile void type;};
template <>          struct add_lvalue_reference<const volatile void> {typedef const volatile void type;};

// make_signed / make_unsigned

struct __nat
{
};

template <class _Hp, class _Tp>
struct __type_list
{
    typedef _Hp _Head;
    typedef _Tp _Tail;
};

typedef
    __type_list<signed char,
    __type_list<signed short,
    __type_list<signed int,
    __type_list<signed long,
    __type_list<signed long long,
    __nat
    > > > > > __signed_types;

typedef
    __type_list<unsigned char,
    __type_list<unsigned short,
    __type_list<unsigned int,
    __type_list<unsigned long,
    __type_list<unsigned long long,
    __nat
    > > > > > __unsigned_types;

template <class _TypeList, size_t _Size, bool = _Size <= sizeof(typename _TypeList::_Head)> struct __find_first;

template <class _Hp, class _Tp, size_t _Size>
struct __find_first<__type_list<_Hp, _Tp>, _Size, true>
{
    typedef _Hp type;
};

template <class _Hp, class _Tp, size_t _Size>
struct __find_first<__type_list<_Hp, _Tp>, _Size, false>
{
    typedef typename __find_first<_Tp, _Size>::type type;
};

template <class _Tp, class _Up, bool = is_const<typename remove_reference<_Tp>::type>::value,
                             bool = is_volatile<typename remove_reference<_Tp>::type>::value>
struct __apply_cv
{
    typedef _Up type;
};

template <class _Tp, class _Up>
struct __apply_cv<_Tp, _Up, true, false>
{
    typedef const _Up type;
};

template <class _Tp, class _Up>
struct __apply_cv<_Tp, _Up, false, true>
{
    typedef volatile _Up type;
};

template <class _Tp, class _Up>
struct __apply_cv<_Tp, _Up, true, true>
{
    typedef const volatile _Up type;
};

template <class _Tp, class _Up>
struct __apply_cv<_Tp&, _Up, false, false>
{
    typedef _Up& type;
};

template <class _Tp, class _Up>
struct __apply_cv<_Tp&, _Up, true, false>
{
    typedef const _Up& type;
};

template <class _Tp, class _Up>
struct __apply_cv<_Tp&, _Up, false, true>
{
    typedef volatile _Up& type;
};

template <class _Tp, class _Up>
struct __apply_cv<_Tp&, _Up, true, true>
{
    typedef const volatile _Up& type;
};

template <class _Tp, bool = is_integral<_Tp>::value || is_enum<_Tp>::value>
struct __make_signed {};

template <class _Tp>
struct __make_signed<_Tp, true>
{
    typedef typename __find_first<__signed_types, sizeof(_Tp)>::type type;
};

template <> struct __make_signed<bool,               true> {};
template <> struct __make_signed<  signed short,     true> {typedef short     type;};
template <> struct __make_signed<unsigned short,     true> {typedef short     type;};
template <> struct __make_signed<  signed int,       true> {typedef int       type;};
template <> struct __make_signed<unsigned int,       true> {typedef int       type;};
template <> struct __make_signed<  signed long,      true> {typedef long      type;};
template <> struct __make_signed<unsigned long,      true> {typedef long      type;};
template <> struct __make_signed<  signed long long, true> {typedef long long type;};
template <> struct __make_signed<unsigned long long, true> {typedef long long type;};

template <class _Tp>
struct make_signed
{
    typedef typename __apply_cv<_Tp, typename __make_signed<typename remove_cv<_Tp>::type>::type>::type type;
};

template <class _Tp, bool = is_integral<_Tp>::value || is_enum<_Tp>::value>
struct __make_unsigned {};

template <class _Tp>
struct __make_unsigned<_Tp, true>
{
    typedef typename __find_first<__unsigned_types, sizeof(_Tp)>::type type;
};

template <> struct __make_unsigned<bool,               true> {};
template <> struct __make_unsigned<  signed short,     true> {typedef unsigned short     type;};
template <> struct __make_unsigned<unsigned short,     true> {typedef unsigned short     type;};
template <> struct __make_unsigned<  signed int,       true> {typedef unsigned int       type;};
template <> struct __make_unsigned<unsigned int,       true> {typedef unsigned int       type;};
template <> struct __make_unsigned<  signed long,      true> {typedef unsigned long      type;};
template <> struct __make_unsigned<unsigned long,      true> {typedef unsigned long      type;};
template <> struct __make_unsigned<  signed long long, true> {typedef unsigned long long type;};
template <> struct __make_unsigned<unsigned long long, true> {typedef unsigned long long type;};

template <class _Tp>
struct make_unsigned
{
    typedef typename __apply_cv<_Tp, typename __make_unsigned<typename remove_cv<_Tp>::type>::type>::type type;
};


}

}


