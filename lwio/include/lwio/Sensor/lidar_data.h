
#pragma once 

#include <unordered_map>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace lwio {
using type_id = uint8_t;

const std::string POINTS_PROCESSED_NAME = "processed"; 
const std::string POINTS_LINE_FEATURE_NAME = "line"; 
const std::string POINTS_PLANE_FEATURE_NAME = "plane"; 

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief:  一个雷达的数据     包含时间戳+数据体
 */    
template<typename _PointT>
struct LidarData {
    double timestamp_ = -1;
    double end_timestamp_ = -1;
    typename pcl::PointCloud<_PointT>::Ptr pointcloud_ptr_{new pcl::PointCloud<_PointT>()};  
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief  多激光点云数据结构 
 * @details 传入到激光估计器的数据  
 */
template<class _PointT>
struct MultiLidarData {
    double timestamp;    // In second.
    std::vector<std::pair<type_id, LidarData<_PointT>>> all_lidar_data;      // <激光id, 点云>
};

//////////////////////////////////////////////////////结果信息发布/////////////////////////////////////////////// 
/**
* @brief:  激光帧信息
* @details 单独一个激光的数据   用与可视化
* @param _PointcloudType 该激光的数据类型   
*/    
template<typename _PointType>
struct LidarResultInfo
{
    double time_stamps_;  
    Eigen::Matrix4f pose_ = Eigen::Matrix4f::Identity();    // 雷达的位姿
    std::unordered_map<std::string, 
        typename pcl::PointCloud<_PointType>::ConstPtr> feature_point_;   // 可视化 点云  
    std::unordered_map<std::string, 
        typename pcl::PointCloud<_PointType>::ConstPtr> local_map_;   // 可视化 局部  
};

// 保存多激光一个帧数据的特征
template<typename _PointType>
using MultiLidarResultInfo
    = std::unordered_map<type_id, LidarResultInfo<_PointType>>;  // <激光的ID，点云>
}
