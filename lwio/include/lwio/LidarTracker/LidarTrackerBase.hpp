#pragma once 

#include <eigen3/Eigen/Dense>
#include "SlamLib/Common/pointcloud.h"

namespace lwio {
/**
 * @brief:  
 * @param _FeatureInfoType tracker 处理的特征数据结构
 */
template<typename _PointType>
class LidarTrackerBase {
public:
    using LocalMapContainer = std::unordered_map<std::string, SlamLib::PCLPtr<_PointType>>;     // Local map 类型 
    virtual ~LidarTrackerBase() {}
    /**
     * @brief: 求解tracker 
     * @param[in] data 用于求解的特征数据
     * @param[out] T 输入预测位姿态, 输出结果
     */        
    virtual void Solve(const SlamLib::CloudContainer<_PointType>& data, 
                                            Eigen::Isometry3d &deltaT) = 0;
    virtual const Eigen::Isometry3d& GetCurrPoseInLocalFrame() const = 0;
    virtual const SlamLib::PCLPtr<_PointType>& GetDynamicCloud() const = 0;  
    virtual const SlamLib::PCLPtr<_PointType>& GetFalseDynamicCloud() const = 0;  
    virtual const LocalMapContainer& GetLocalMap() const = 0;  
    virtual bool HasUpdataLocalMap() const = 0;
    virtual void ResetLocalmap() = 0;
};
} // class LidarTrackerBase 


