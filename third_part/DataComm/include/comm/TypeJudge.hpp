/**
 * @file DataDispatcher.hpp
 * @brief 线程间通信
 * @author lwh
 * @version 1.0
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#pragma once 
#include <iostream>
namespace  comm {
template<typename _T>
class is_const {
public:
    static const uint8_t value = 0; 
};

template<typename _T>
class is_const<const _T> {
public:
    static const uint8_t value = 1; 
};

template<typename _T>
class is_const<const _T&> {
public:
    static const uint8_t value = 1; 
};

enum class ref_type {non_ref = 0, lvalue_ref, rvalue_ref};

template<typename _T>
class is_ref {
public:
    static ref_type is() {
        return ref_type::non_ref;  
    }
};

template<typename _T>
class is_ref<_T&> {
public:
    static ref_type is() {
        return ref_type::lvalue_ref;  
    }
};

template<typename _T>
class is_ref<_T&&> {
public:
    static ref_type is() {
        return ref_type::rvalue_ref;  
    }
};
}