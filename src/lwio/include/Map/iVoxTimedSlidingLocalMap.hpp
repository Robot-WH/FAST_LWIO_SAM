
#pragma once

#include "LocalMap.hpp"
#include "Algorithm/PointClouds/Map/ivox3d/ivox3d.h"  

namespace Slam3D {
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief:  ivox + 时间滑动窗口 
     * @param _PointType local map中每个地图点的类型  
     */    
    template<typename _PointType>
    class IvoxTimedSlidingLocalMap : public PointCloudLocalMapBase<_PointType> {
        protected:
            using base = PointCloudLocalMapBase<_PointType>;
            using PointCloudConstPtr = typename pcl::PointCloud<_PointType>::ConstPtr;
            using PointVector = typename base::PointVector; 
            using KdtreePtr = typename pcl::KdTreeFLANN<_PointType>::Ptr; 
        public:
            using IvoxNearbyType = typename IVox<_PointType>::NearbyType;
            struct  Option {
                int window_size_ = 10; 
                typename IVox<_PointType>::Options ivox_option_;  
            };

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////
            IvoxTimedSlidingLocalMap(Option option) : option_(option) {
                LOG(INFO) << "create IvoxTimedSlidingLocalMap"; 
            }

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////
            /**
             * @brief: 由于发生了充足的运动，因此添加一帧数据到Local map   
             */            
            void UpdateLocalMapForMotion(std::string const& name, 
                    typename pcl::PointCloud<_PointType>::ConstPtr const& frame,
                    std::vector<PointVector> const& nearest_points) override {
                if (frame->empty()) return;  
                std::lock_guard<std::mutex> lock(base::local_map_mt_);
                if (local_map_frame_container_.find(name) == local_map_frame_container_.end()) {
                    base::local_map_container_[name].reset(new pcl::PointCloud<_PointType>()); 
                    ivox_container_.insert(std::make_pair(name, IVox<_PointType>(option_.ivox_option_)));
                }
                auto& frame_queue_ = local_map_frame_container_[name];
                typename pcl::PointCloud<_PointType>::ConstPtr delete_frame;  
                // 更新滑动窗口  
                if (frame_queue_.size() >= option_.window_size_) {  
                    base::full_ = true;   
                    delete_frame = frame_queue_.front();  
                    frame_queue_.pop_front();
                    frame_queue_.push_back(frame);
                    base::local_map_container_[name]->clear();   
                    // 更新submap  
                    for (typename std::deque<PointCloudConstPtr>::const_iterator it = frame_queue_.begin(); 
                        it != frame_queue_.end(); it++) {
                            *base::local_map_container_[name] += **it;   
                    }
                    // ivox删除最老帧
                    ivox_container_.at(name).DeletePoints(*delete_frame);
                } else {  
                    *base::local_map_container_[name] += *frame;      
                    frame_queue_.push_back(frame);   
                }
                // ivox添加最新帧  
                ivox_container_.at(name).AddPoints(*frame);
                // std::cout<<"map_name size: "
                // <<base::local_map_.second->size()<<std::endl;
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
                    base::local_map_container_[name].reset(new pcl::PointCloud<_PointType>()); 
                    ivox_container_.insert(std::make_pair(name, IVox<_PointType>(option_.ivox_option_)));
                }
                auto& frame_queue_ = local_map_frame_container_[name];
                if (!frame_queue_.empty()) {
                    typename pcl::PointCloud<_PointType>::ConstPtr delete_frame;  
                    delete_frame = frame_queue_.back();  
                    frame_queue_.pop_back();
                    ivox_container_.at(name).DeletePoints(*delete_frame);
                }
                frame_queue_.push_back(frame);
                ivox_container_.at(name).AddPoints(*frame);
                // 更新submap  
                base::local_map_container_[name]->clear();   
                for (typename std::deque<PointCloudConstPtr>::const_iterator it = frame_queue_.begin(); 
                    it != frame_queue_.end(); it++) {
                    *base::local_map_container_[name] += **it;   
                }
                return;  
            }
            
            /**
             * @brief: 搜索localmap
             * @return 是否搜索到 num 个点 
             */            
            virtual bool GetNearlyNeighbor(std::string const& name, _PointType const& point, 
                    uint16_t const& num, double const& max_range, PointVector& res) const override {
                if (ivox_container_.find(name) == ivox_container_.end()) return false;   
                // ivox 中 num 表示最多提取的点的数量，而我们这里标识的是最少提取点的数量   
                //TicToc tt;
                ivox_container_.at(name).GetClosestPoint(point, res, num, max_range); 
                if (res.size() < num) {
                    return false; 
                }    
                //tt.toc("ivox knn ");
                return true;  
            } 

        private:
            Option option_; 
            std::unordered_map<std::string, IVox<_PointType>> ivox_container_;  
            std::unordered_map<std::string, 
                std::deque<typename pcl::PointCloud<_PointType>::ConstPtr>> local_map_frame_container_;  
    }; // class IvoxTimedSlidingLocalMap
}

