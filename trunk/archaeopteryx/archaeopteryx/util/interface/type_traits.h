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

template <bool, class _Tp> struct enable_if {};
template <class _Tp> struct enable_if<true, _Tp> {typedef _Tp type;};

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


// add_lvalue_reference

template <class _Tp> struct add_lvalue_reference {typedef _Tp& type;};
//template <class _Tp> struct add_lvalue_reference<_Tp&> {typedef _Tp& type;};  // for older compiler
template <>          struct add_lvalue_reference<void> {typedef void type;};
template <>          struct add_lvalue_reference<const void> {typedef const void type;};
template <>          struct add_lvalue_reference<volatile void> {typedef volatile void type;};
template <>          struct add_lvalue_reference<const volatile void> {typedef const volatile void type;};

}

}


