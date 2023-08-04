
#pragma once 
#define PCL_NO_PRECOMPILE
#include <execution>  // C++ 17 并行算法 
#include <atomic>
#include <shared_mutex>
#include <omp.h>  
#include <yaml-cpp/yaml.h>
#include "LidarTracker/LidarTracker.hpp"
#include "Common/pose.hpp"
#include "Common/color.hpp"
#include "Common/keyframe_updater.hpp"
#include "Common/utility.hpp"
#include "Initialized/imu_initialized.hpp"
#include "lwio/LidarProcess/Segmentation/PointCloudSegmentation.hpp"
#include "lwio/LidarProcess/GroundDetect/DirectGroundDetect.hpp"
#include "lwio/Estimator/calibration/handeye_calibration_base.hpp"
#include "comm/InnerProcessComm.hpp"
#include "Sensor/lidar_data.h"
#include "LidarProcess/Preprocess/LidarPreProcess.hpp"

namespace lwio {

///////////////////////////////////////////////////////////////////////////////////////////////////// 
/**
* @brief:  估计结果信息发布
* @details 单独一个激光的数据   用与可视化
* @param _PointcloudType 该激光的数据类型   
*/    
template<typename _PointT>
struct ResultInfo {
    double time_stamps_;  
    Eigen::Isometry3d pose_ = Eigen::Isometry3d::Identity();    // 雷达的位姿
    SlamLib::FeaturePointCloudContainer<_PointT> pointcloud_;   // 可视化 点云  
};
    
template<typename _PointT>
class System {
public:
    using LidarTrackerPtr = std::unique_ptr<LidarTrackerBase<_PointT>>;  
private:
    enum class SensorType {LIDAR, IMU, GNSS};   // 默认等于枚举的第一个元素 
    struct OdomInfo {
        Eigen::Isometry3d motion_ = Eigen::Isometry3d::Identity();
        double time_begin_ = 0.;
        double time_end_ = 0.;
    };
public:
    System(std::string param_path);
    void InputData(LidarData<_PointT>& curr_data);
    void InputData(sensor::ImuData& data);
    void InputData(sensor::GnssData& data);
    void InputData(sensor::WheelOdom& data);
private:
    void processMeasurements();
    bool syncSensorData(const double& start_time, const double& end_time,
                                            std::deque<sensor::WheelOdom>& wheelOdom_container, 
                                            std::deque<sensor::ImuData>& imu_container);

    template<class DataT_>
    bool extractSensorData(std::deque<DataT_>& data_cache, std::deque<DataT_>& extracted_container,
            const double& start_time, const double& end_time);

private:
    std::string param_path_;    // 参数配置地址 
    // 模块  
    std::unique_ptr<LidarPreProcess<_PointT>> preprocess_;    
    LidarTrackerPtr lidar_trackers_;  
    ImuInitialize imu_init_;  
    // std::unique_ptr<PointCloudSegmentation<_PointT>> segmentation_;  
    typename DirectGroundDetect<_PointT>::Ptr ground_detect_;    // 地面检测
    // std::unique_ptr<LIGEstimatorInterface<_PointT>> estimator_;    // 估计器  
    Eigen::Isometry3d pose_;    // 运动
    HandEyeCalibrationBase extrinsics_init_;    // 外参初始化
    KeyframeUpdater keyframe_updater_; 

    std::thread process_thread_;
    // 提取的特征数据       <时间戳，vector(每一个雷达的特征数据)>  
    std::deque<SlamLib::CloudContainer<_PointT>> processed_lidar_buf_;  
    std::deque<sensor::ImuData> imu_buf_;    
    std::deque<sensor::ImuData> imu_init_buf_;    // 保留
    std::deque<sensor::GnssData> gnss_buf_;     
    std::deque<sensor::WheelOdom> odom_buf_;   

    bool SYSTEM_INIT_ = false;   
    bool INS_INIT_ = false;  
    bool EXTRINSIC_GET_ = true;   
    bool EXTRINSIC_OPTI_EN_ = true;  //  外参在线优化 标志
    std::shared_mutex wheel_sm_;  
    std::shared_mutex imu_sm_;  
    std::shared_mutex lidar_sm_;  
    std::shared_mutex gnss_sm_;  
    std::shared_mutex odom_sm_;  
};
}// namespace 


