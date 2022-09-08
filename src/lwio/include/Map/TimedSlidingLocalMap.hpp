
#pragma once

#include "LocalMap.hpp"
#include <pcl/kdtree/kdtree_flann.h>

namespace Slam3D {
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief:  基于关键帧滑动窗口的 '地图点数据 localmap'  
     * @param _PointType local map中每个地图点的类型  
     */    
    template<typename _PointType>
    class TimedSlidingLocalMap : public PointCloudLocalMapBase<_PointType> {
        protected:
            using Base = PointCloudLocalMapBase<_PointType>;
            using PointCloudConstPtr = typename pcl::PointCloud<_PointType>::ConstPtr;
            using PointVector = typename Base::PointVector; 
            using KdtreePtr = typename pcl::KdTreeFLANN<_PointType>::Ptr; 
        public:
            struct  Option {
                int window_size_ = 10; 
                bool use_kdtree_search_ = true;     // 是否使用kdtree
            };

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////
            TimedSlidingLocalMap(Option option) : option_(option) {
                LOG(INFO) << "create TimedSlidingLocalMap, window size:" << option_.window_size_
                <<", kdtree_enable: "<<option_.use_kdtree_search_;
            }

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////
            /**
             * @brief: 由于发生了充足的运动，因此添加一帧数据到Local map   
             */            
            void UpdateLocalMapForMotion(std::string const& name, 
                    typename pcl::PointCloud<_PointType>::ConstPtr const& frame,
                    std::vector<PointVector> const& nearest_points) override {
                if (frame->empty()) return;  
                std::lock_guard<std::mutex> lock(Base::local_map_mt_);
                if (local_map_frame_container_.find(name) == local_map_frame_container_.end()) {
                    Base::local_map_container_[name].reset(new pcl::PointCloud<_PointType>()); 
                    kdtree_container_[name].reset(new pcl::KdTreeFLANN<_PointType>()); 
                }
                auto& frame_queue_ = local_map_frame_container_[name];
                //TicToc tt;
                // 更新滑动窗口      0.1ms   
                if (frame_queue_.size() >= option_.window_size_) {  
                    Base::full_ = true;   
                    frame_queue_.pop_front();
                    frame_queue_.push_back(frame);
                    Base::local_map_container_[name]->clear();   
                    // 更新submap  
                    for (typename std::deque<PointCloudConstPtr>::const_iterator it = frame_queue_.begin(); 
                        it != frame_queue_.end(); it++) {
                            *Base::local_map_container_[name] += **it;   
                    }
                } else {  
                    *Base::local_map_container_[name] += *frame;      
                    frame_queue_.push_back(frame);   
                }
                // tt.toc("update localmap points ");
                // tt.tic();
                // 耗时 > 5ms  
                if (option_.use_kdtree_search_) {
                    kdtree_container_[name]->setInputCloud(Base::local_map_container_[name]);
                }
                //tt.toc("local map kdtree ");
                // std::cout<<"map_name size: "
                // <<Base::local_map_.second->size()<<std::endl;
                return;  
            }

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////
            /**
             * @brief: 由于长时间没有更新地图，因此添加一帧数据到Local map   
             * @details 此时直接将滑动窗口最末尾的数据移除
             */            
            void UpdateLocalMapForTime(std::string const& name, 
                    typename pcl::PointCloud<_PointType>::ConstPtr const& frame,
                    std::vector<PointVector> const& nearest_points) override {
                if (frame->empty()) return;  
                if (local_map_frame_container_.find(name) == local_map_frame_container_.end()) {
                    Base::local_map_container_[name].reset(new pcl::PointCloud<_PointType>()); 
                    kdtree_container_[name].reset(new pcl::KdTreeFLANN<_PointType>()); 
                }
                auto& frame_queue_ = local_map_frame_container_[name];
                if (!frame_queue_.empty()) {
                    frame_queue_.pop_back();
                }
                frame_queue_.push_back(frame);
                // 更新submap  
                Base::local_map_container_[name]->clear();   
                for (typename std::deque<PointCloudConstPtr>::const_iterator it = frame_queue_.begin(); 
                    it != frame_queue_.end(); it++) {
                    *Base::local_map_container_[name] += **it;   
                }
                if (option_.use_kdtree_search_) {
                    kdtree_container_[name]->setInputCloud(Base::local_map_container_[name]);
                }
                return;  
            }
            
            /**
             * @brief: 找到 point 的近邻点
             * @param name 点的标识名
             * @param num 需要的近邻点数量
             * @param max_range 近邻点的最大距离
             * @param[out] res 查找的结果 
             * @return 是否搜索到 num 个点 
             */            
            virtual bool GetNearlyNeighbor(std::string const& name, _PointType const& point, 
                    uint16_t const& num, double const& max_range, PointVector& res) const override {
                if (!option_.use_kdtree_search_) return false;  
                std::vector<int> pointSearchInd;
                std::vector<float> pointSearchSqDis;
                //TicToc tt;
                kdtree_container_.at(name)->nearestKSearch(point, num, pointSearchInd, pointSearchSqDis); 
                //tt.toc("kdtree knn ");
                double sq_max_range = max_range * max_range;
                res.clear();
                for (uint16_t i = 0; i < pointSearchInd.size(); i++) {
                    if (pointSearchSqDis[i] > sq_max_range) break; 
                    res.emplace_back(Base::local_map_container_.at(name)->points[pointSearchInd[i]]);
                }
                if (res.size() < num) return false;
                return true;  
            } 

        private:
            Option option_; 
            std::unordered_map<std::string, KdtreePtr> kdtree_container_;  
            std::unordered_map<std::string, 
                std::deque<typename pcl::PointCloud<_PointType>::ConstPtr>> local_map_frame_container_;  
    }; // class TimedSlidingLocalMap
}

