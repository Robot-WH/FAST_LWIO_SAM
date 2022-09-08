
#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace Slam3D {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief:  基于 地图点数据的 localmap  
 * @details:  local map 随着使用的地图点类型的变化而变化 
 * @param _PointType local map中每个地图点的类型  ,地图点包含的信息通过这个进行改变
 */    
template<typename _PointType>
class PointCloudLocalMapBase {
    protected:
            // 地图名字 + 数据
        using LocalMapContainer = std::unordered_map<std::string, 
            typename pcl::PointCloud<_PointType>::Ptr>;
        using ConstLocalMapContainer = std::unordered_map<std::string, 
            typename pcl::PointCloud<_PointType>::ConstPtr>;
        using LocalMapType = typename pcl::PointCloud<_PointType>::Ptr; 
    public:
        using Ptr = std::shared_ptr<PointCloudLocalMapBase<_PointType>>; 
        using ConstPtr = std::shared_ptr<const PointCloudLocalMapBase<_PointType>>; 
        using PointVector = std::vector<_PointType, Eigen::aligned_allocator<_PointType>>;
        PointCloudLocalMapBase() {}

        virtual ~PointCloudLocalMapBase() {}
        virtual void UpdateLocalMapForMotion(std::string const& name, 
            typename pcl::PointCloud<_PointType>::ConstPtr const& frame,
            std::vector<PointVector> const& nearest_points) = 0; 
        virtual void UpdateLocalMapForTime(std::string const& name, 
            typename pcl::PointCloud<_PointType>::ConstPtr const& frame,
            std::vector<PointVector> const& nearest_points) = 0; 
       virtual bool GetNearlyNeighbor(std::string const& name, _PointType const& point, 
            uint16_t const& num, double const& max_range, PointVector& res) const = 0;   // 搜索邻居  

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 获取名字为name 的local map 
         * @param name
         * @return 点云类型 
         */            
        LocalMapType GetLocalMap(std::string const& name) {
            std::lock_guard<std::mutex> lock(local_map_mt_);
            if (local_map_container_.find(name) == local_map_container_.end()) {
                return nullptr; 
            }
            return local_map_container_[name];
        }

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 获取全部的local map 
         * @details 重载  
         * @return map<name, 点云>
         */     
        ConstLocalMapContainer GetLocalMap() {
            std::lock_guard<std::mutex> lock(local_map_mt_);
            ConstLocalMapContainer const_container;
            for (auto const& it : local_map_container_) {
                const_container.insert(it);
            }
            return const_container; 
        }

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // 返回local map 队列是否满  
        bool is_full() const {
            return full_;
        }

    protected:
        std::mutex local_map_mt_; 
        LocalMapContainer local_map_container_;  
        bool full_ = false;  
};
}
