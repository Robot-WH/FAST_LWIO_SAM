#pragma once 

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>

namespace Slam3D {
namespace Algorithm {
namespace Filter {
/**
 * @brief: pcl 滤波器的封装
 * @details: 
 * @return {*}
 */
template<typename _PointType>
class FilterBase {
    public:
        using PointCloudPtr = typename pcl::PointCloud<_PointType>::Ptr;  
        using PointCloudConstPtr = typename pcl::PointCloud<_PointType>::ConstPtr; 

        FilterBase() : filter_ptr_(nullptr) {}
        virtual ~FilterBase() {}
        /**
         * @brief:  滤波流程
         * @param[in] cloud_in 输入的点云 
         * @param[out] cloud_out 处理后的点云 
         */        
        virtual PointCloudPtr Filter(const PointCloudConstPtr &cloud_in) const {
            PointCloudPtr cloud_out(
                new pcl::PointCloud<_PointType>(*cloud_in));  
            if (filter_ptr_ == nullptr)
                return cloud_out; 
            filter_ptr_->setInputCloud(cloud_in);
            filter_ptr_->filter(*cloud_out);
            cloud_out->header = cloud_in->header;
            return cloud_out;  
        }

        /**
         * @brief: 设置滤波器   通过这个就能更换滤波器
         */        
        void SetFilter(typename pcl::Filter<_PointType>::Ptr const& filter_ptr) {
            filter_ptr_ = filter_ptr;   
        }
        
    private:
        typename pcl::Filter<_PointType>::Ptr filter_ptr_; 
}; // class FilterBase
}
}
}
