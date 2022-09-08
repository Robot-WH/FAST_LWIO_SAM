#pragma once 

#include <eigen3/Eigen/Dense>
#include "Sensor/lidar_data_type.h"

namespace Slam3D {
/**
 * @brief:  
 * @param _FeatureInfoType tracker 处理的特征数据结构
 */
template<typename _PointType>
class LidarTrackerBase {
    private:
        using PointCloudPtr = typename pcl::PointCloud<_PointType>::Ptr;  
        using PointCloudConstPtr = typename pcl::PointCloud<_PointType>::ConstPtr;  
    public:
        virtual ~LidarTrackerBase() {}
        /**
         * @brief: 求解tracker 
         * @param[in] data 用于求解的特征数据
         * @param[out] T 输入预测位姿态, 输出结果
         */        
        virtual void Solve(CloudContainer<_PointType> const& data, Eigen::Isometry3d &deltaT) = 0;
        virtual Eigen::Isometry3d const& GetCurrPoseInLocalFrame() const = 0;
        //virtual std::unordered_map<std::string, PointCloudConstPtr> GetLocalMap() const = 0; 
};
} // class LidarTrackerBase 


