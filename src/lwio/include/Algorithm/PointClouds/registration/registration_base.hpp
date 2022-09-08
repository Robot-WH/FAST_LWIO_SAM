#pragma once 

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <eigen3/Eigen/Dense>
#include "Sensor/lidar_data_type.h"
#include "Map/LocalMap.hpp"

namespace Slam3D {
namespace Algorithm {

template<typename _PointType>
class RegistrationBase {
    public:
        using LocalMapConstPtr = typename PointCloudLocalMapBase<_PointType>::ConstPtr;  
        using PointCloudConstPtr = typename pcl::PointCloud<_PointType>::ConstPtr;  
        using PointVector = std::vector<_PointType, Eigen::aligned_allocator<_PointType>>;
        using Ptr = std::unique_ptr<RegistrationBase<_PointType>>;
        // 每个点匹配后的信息
        struct pointRegistrationResult {
            std::vector<double> residuals_;   // 匹配的残差
            std::vector<PointVector> nearly_points_;    // 邻居点 
        };  
        using RegistrationResult = std::unordered_map<std::string, pointRegistrationResult>;

        virtual void SetInputSource(FeaturePointCloudContainer<_PointType> const& source_input) = 0;  
        virtual void SetInputTarget(FeaturePointCloudContainer<_PointType> const& target_input)  = 0;  
        virtual void SetInputTarget(LocalMapConstPtr const& target_input) = 0;
        virtual void SetMaxIteration(uint16_t const& n) = 0;
        virtual void SetNormIteration(uint16_t const& n) = 0; 
        virtual void Solve(Eigen::Isometry3d &T) = 0; 
        virtual RegistrationResult const& GetRegistrationResult() const = 0;
}; // class LineSurfFeatureRegistration 
} // namespace Algorithm
}
