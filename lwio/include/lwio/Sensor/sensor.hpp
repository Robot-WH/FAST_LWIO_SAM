/**
 * @file sensor.hpp
 * @author lwh ()
 * @brief 
 * @version 0.1
 * @date 2023-06-08
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once

#include <eigen3/Eigen/Dense>
namespace lwio {
namespace sensor { 

enum sensor_id {lidar, imu, gnss, wheel};  

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief IMU数据结构 
 */
struct ImuData {
    ImuData() {}
    ImuData(double const& timestamp, Eigen::Vector3d const& acc, 
        Eigen::Vector3d const& gyro, Eigen::Quaterniond const& rot) 
    : timestamp_(timestamp), acc_(acc), gyro_(gyro), rot_(rot) {}
    double timestamp_ = -1;      // In second.
    Eigen::Vector3d acc_ = {0,0,0};   // Acceleration in m/s^2
    Eigen::Vector3d gyro_ = {0,0,0};  // Angular velocity in radian/s.
    Eigen::Quaterniond rot_ = Eigen::Quaterniond::Identity(); // 旋转  
};

using ImuDataPtr = std::shared_ptr<ImuData>; 
using ImuDataConstPtr = std::shared_ptr<const ImuData>; 

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief GNSS数据结构 
 */
struct GnssData {
    double timestamp_;      // In second.
    // WGS84系 
    Eigen::Vector3d lla_; 
    Eigen::Matrix3d cov_;  // Covariance in m^2. 
};

using GnssDataPtr = std::shared_ptr<GnssData>;
using GnssDataConstPtr = std::shared_ptr<const GnssData>;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief 轮速 数据结构 
 */
struct WheelsData {
};

/**
 * @brief: 里程计数据  
 * @details 轮式里程计为2D
 */    
struct OdomData {
    double timestamp_ = -1;      // In second.
    float velocity_ = 0.0f;  // 线速度
    float yaw_angular_vel_ = 0.0f;  // yaw角速度 
};

}
} // namespace Sensor 