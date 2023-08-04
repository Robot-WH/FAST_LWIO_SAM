
#pragma once 
#include <iostream>

namespace common {

/**
 * @brief: 引用移除  
 */    
template<typename T>
struct remove_reference {
    public:
        typedef T type;  
};   

template<typename T>
struct remove_reference<T&> {
    public:
        typedef T type;  
};   

template<typename T>
struct remove_reference<T&&> {
    public:
        typedef T type;  
};   

template<class T>
using remove_reference_t = typename remove_reference<T>::type;  

};  // namespace common
