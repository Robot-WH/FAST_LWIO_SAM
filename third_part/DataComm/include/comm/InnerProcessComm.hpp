/**
 * @brief 线程间通信
 * @author lwh
 * @version 1.0
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#pragma once 
#include <set>
#include <list>
#include <deque>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <condition_variable>
#include <typeindex>
#include <type_traits>
#include <atomic>
#include "TypeJudge.hpp"
// #include "Function.hpp"
// 进程内部的通信  
namespace comm {
namespace IntraProcess {
/**
 * @brief: 数据管理器对外接口类  
 * @details:  非模板的万能类  
 */    
class CallBackWrapperBase {
public:
    CallBackWrapperBase(bool is_const = false) : is_const_(is_const) {}
    virtual ~CallBackWrapperBase() {}
    virtual std::type_index GetDataType() const = 0; 
    const bool& IsConstParam() const {return is_const_;}
protected:
    bool is_const_;
};

/**
 * @brief: 数据管理器 实现
 */
template<typename _InputT>
class CallBackWrapper : public CallBackWrapperBase {
public:
    CallBackWrapper(std::function<void(_InputT)> call_back) 
    : CallBackWrapperBase(is_const<_InputT>::value), call_back_(call_back), 
        type_info_(typeid(_InputT))// typeid 不区分const和&  也就是 const int& 和 int 是一样的
    {}
    
    ~CallBackWrapper() {}

    /**
     * @brief: 传入引用
     */    
    template<typename _T>
    void Call(_T& data) {
        call_back_(data);
    }

    // 获取数据类型  
    std::type_index GetDataType() const override {
        return type_info_;  
    }
    
private:
    std::mutex data_m_;   // 加了mutex后变成了只移型别   (禁止了拷贝构造)
    std::type_index type_info_;  // 数据类型信息 
    std::function<void(_InputT)> call_back_;   
};

/**
 * @brief: 订阅者类   
 * @details 包含订阅回调函数和订阅数据缓存
 *                      由于在DataDispatcher类中通过一个容器管理所有Topic的Subscriber，因此该类必须是非模板基类
 */    
class Subscriber {
public:
    Subscriber() {}

    /**
     * @brief: 构造函数
     * @param callback 类内成员函数地址,只能绑定 const&类型的函数
     * @param class_addr 类对象地址
     *  std::function的模板参数不一定要和std::bind()的callback参数一样，
     *  因为std::function调用callback的逻辑是，std::function的operator()->binder_wrapper_->Call()
     * std::function的模板参数只决定operator()
     */        
    template<typename _InputT, typename _Ctype>
    Subscriber(void (_Ctype::*callback)(_InputT), _Ctype* class_addr) {
        static_assert(std::is_lvalue_reference<_InputT>::value, 
            "Error: callback parameter must be a lvalue_reference type");
        callback_wrap_base_ = new CallBackWrapper<_InputT>(
            // std::bind(callback, class_addr, std::placeholders::_1)
            [callback, class_addr](_InputT input) {
                (class_addr->*callback)(input);
            }
        );
    }

    /**
     * @brief: 构造函数
     * @param callback 普通成员函数地址
     * @param  cache_capacity 缓存容量  
     * @param high_priority 是否为高优先级
     */      
    template<typename _InputT>
    Subscriber(void (*callback)(_InputT)) {
        static_assert(std::is_lvalue_reference<_InputT>::value, 
            "Error: callback parameter must be a lvalue_reference type");
        callback_wrap_base_ = new CallBackWrapper<_InputT>(
            [callback](_InputT input) {
                (*callback)(input);
            }
        );
    }

    virtual ~Subscriber() {
        // std::cout << "~Subscriber" <<std::endl;
        delete callback_wrap_base_; 
    }

    /**
     * @brief: 在DataDispatcher类的Publish函数中被调用
     */    
    template<typename _DataT>
    void send(_DataT& data) {
        send_m_.lock();   // 避免同时执行多个

        if (callback_wrap_base_->IsConstParam()) {
            // 回调函数的参数是const的  
            auto callback_wrap =  
                dynamic_cast<CallBackWrapper<const _DataT&>*>(callback_wrap_base_);
            if (callback_wrap == nullptr) {
                throw std::bad_cast();  
            }
            callback_wrap->Call(data);  
        } else {
            // 回调函数的参数不是const的  
            auto callback_wrap =  
                dynamic_cast<CallBackWrapper<_DataT&>*>(callback_wrap_base_);
            if (callback_wrap == nullptr) {
                throw std::bad_cast();  
            }
            callback_wrap->Call(data);  
        }

        send_m_.unlock();  
    }

    /**
     * @brief 重载
     */
    template<typename _DataT>
    void send(const _DataT& data) {
        send_m_.lock();   // 避免同时执行多个
        auto callback_wrap =  
            dynamic_cast<CallBackWrapper<const _DataT&>*>(callback_wrap_base_);
        if (callback_wrap == nullptr) {
            throw std::bad_cast();  
        }
        callback_wrap->Call(data);  
        send_m_.unlock();  
    }
private:
    // 线程相关
    std::mutex send_m_; 
    // friend class Server; 
    // 数据管理 (数据缓存与回调调度)    要求能处理任何的数据，因此必须是非模板泛化基类
    CallBackWrapperBase* callback_wrap_base_ = nullptr;  
};

/**
 * @brief: 进程内部通信
 * @details:  负责进程内数据传输
 */    
class Server {
public:
    /**
     * @brief: 单例的创建函数  
     */            
    static Server& Instance() {
        static Server server;
        return server; 
    }

    virtual ~Server() {
        for (const auto& name_set : subscriber_container_) {
            for (const auto& pt : name_set.second) {
                delete pt;  
            }
        }
    }
    
    /**
     * @brief: 订阅某个数据容器，回调函数为类内成员的重载 
     * @details: 订阅动作，告知DataDispatcher，_Ctype类对象的callback函数要订阅名字为name的数据容器
     * @param _DataT 回调函数的输入类型  
     * @param name 数据容器名
     * @param callback 注册的回调函数
     * @param class_addr 类对象地址
     * @param cache_capacity 缓存容量 
     * @param high_priority true 高优先级的订阅者在数据发布时会直接调用回调
     *                                                false 低优先级的订阅者在数据发布后由回调线程统一进行调度
     */        
    template<typename _InputT, typename _Ctype>
    Subscriber& Subscribe(std::string const& name, 
                                                        void (_Ctype::*callback)(_InputT), 
                                                        _Ctype* class_addr) {
        Subscriber* p_subscriber = new Subscriber(callback, class_addr);
        substriber_container_m_.lock();
        subscriber_container_[name].insert(p_subscriber); 
        substriber_container_m_.unlock();  
        return *p_subscriber;  
    }

    /**
     * @brief: 订阅某个数据容器，回调函数为普通函数 
     * @param name 数据容器名
     * @param callback 注册的回调函数
     * @param cache_capacity 缓存容量 
     */        
    template<typename _InputT, typename _Ctype>
    Subscriber& Subscribe(std::string const& name, 
                                                    void (*callback)(_InputT)) {
        Subscriber* p_subscriber = new Subscriber(callback);
        substriber_container_m_.lock();
        subscriber_container_[name].insert(p_subscriber); 
        substriber_container_m_.unlock();  
        return *p_subscriber;  
    }

    /**
     * @brief: 发布数据   
     * @details 向数据容器发布数据 done 
     * @param[in] name 数据的标识名
     * @param[in] data 只能发布左值   如果传入的data const T, 则_T推导为const T  
     */            
    template<typename _T>
    void Publish(std::string const& name, _T& data) {
        // 如果有订阅者  则将数据传送到各个订阅者
        std::shared_lock<std::shared_mutex> m_l(substriber_container_m_);  // 禁止subscriber_container_ 写数据
        if (subscriber_container_[name].size()) {
            ///////////////////////////////////////////////////
            // 遍历该topic的所有订阅者，并将数据发送给他们
            for (const auto& pt : subscriber_container_[name]) {
                pt->send(data);     // send是线程安全的
            }
        }
        return;
    }

protected:
    Server() {}
    Server(Server const& object) = default;
    Server(Server&& object) = default; 

private:
    std::shared_mutex substriber_container_m_; 
    std::unordered_map<std::string, std::set<Subscriber*>> subscriber_container_;    // 管理每个Topic 的 Subscriber 
}; // class 
} // namespace 
}