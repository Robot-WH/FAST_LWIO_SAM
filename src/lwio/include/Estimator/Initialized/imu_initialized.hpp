
#ifndef _IMU_INITIALIZED_TOOL_HPP_
#define _IMU_INITIALIZED_TOOL_HPP_

#include <eigen3/Eigen/Dense>
#include "Sensor/gnss_data_process.hpp"
#include "Sensor/sensor.hpp"

namespace Slam3D {

/**
 * @brief: 初始化中IMU的相关操作 
 * @details: 
 */    
class ImuInitialize {
public:
    ImuInitialize() : MeanAccLimit_(0.05),  minimum_dataSize_(100) {}
    ImuInitialize(double MeanAccLimit, int minimum_dataSize) 
        : MeanAccLimit_(MeanAccLimit), minimum_dataSize_(minimum_dataSize) {}

    virtual ~ImuInitialize(){}
    
    /**
     * @brief 检查用于初始化的IMU数据数量
     */
    bool CheckInitializedImuDateNumber(std::vector<ImuDataConstPtr> const& imu_datas) const {
        std::cout<<" imu_datas.size(): " << imu_datas.size() <<" minimum_dataSize_: "
            << minimum_dataSize_ <<std::endl;
        return (imu_datas.size() >= minimum_dataSize_);
    }

    /**
     * @brief 获取静止的IMU平均数据   
     * @details 对静止初始化的IMU数据去平均值     
     * @param [in] acc_buf 加速度容器
     * @param [in] gyro_buf 角速度容器
     * @param [in] rot_buf 旋转四元数容器
     * @param[out] mean_acc 输出平滑后的加速度
     * @param[out] mean_gyro 输出平滑后的角速度 
     * @return 是否满足要求 
     */
    bool CalculateStaticImuMeanData(std::vector<ImuDataConstPtr> cache_imu, 
            Eigen::Vector3d &mean_acc, Eigen::Vector3d &mean_gyro, 
            Eigen::Quaterniond &mean_rot) const {
        // Compute mean and std of the imu buffer.
        Eigen::Vector3d sum_acc(0., 0., 0.);
        Eigen::Vector3d sum_gyro(0., 0., 0.);
        // 计算均值 
        for  (auto const& imu_data : cache_imu) {
            sum_acc += imu_data->acc;
            sum_gyro += imu_data->gyro;
        }
        mean_acc = sum_acc / (double)cache_imu.size();
        mean_gyro = sum_gyro / (double)cache_imu.size();
        Eigen::Vector3d sum_err2(0., 0., 0.);
        // 计算加速度标准差
        for  (const auto imu_data : cache_imu) {
            sum_err2 += (imu_data->acc - mean_acc).cwiseAbs2();
        }
        const Eigen::Vector3d sigma_acc = (sum_err2 / (double)cache_imu.size()).cwiseSqrt();
        // 求模长 检查是否运动过于剧烈 
        if  (sigma_acc.norm() > MeanAccLimit_) {
            std::cout << "[CalculateStaticImuMeanData]: Too big sigma_acc: " 
            << sigma_acc.transpose() << std::endl;
            return false;
        }
        return true;  
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief 通过IMU的重力数据恢复旋转  
     * @param Rwi imu->world的旋转  
     * @param smooth_gravity 平滑后的重力 
     * @return 是否计算成功 
     */
    Eigen::Matrix3d const ComputeRotationFromGravity(Eigen::Vector3d const& smooth_gravity) 
    {
        //  z-axis.   world系 即导航系z轴(重力方向) 在 载体系的表示 
        const Eigen::Vector3d& z_axis = smooth_gravity.normalized(); 
        //  x-axis.
        Eigen::Vector3d x_axis = 
            Eigen::Vector3d::UnitX() - z_axis * z_axis.transpose() * Eigen::Vector3d::UnitX();
        x_axis.normalize();
        // y-axis.
        Eigen::Vector3d y_axis = z_axis.cross(x_axis);
        y_axis.normalize();
        // world -> body
        Eigen::Matrix3d R_i_n;
        R_i_n.block<3, 1>(0, 0) = x_axis;
        R_i_n.block<3, 1>(0, 1) = y_axis;
        R_i_n.block<3, 1>(0, 2) = z_axis;
        // 返回 imu->导航系的旋转 
        return R_i_n.transpose();
    }

private:
    // 初始化加速度阈值   
    double MeanAccLimit_;  
    // IMU缓存数量阈值
    int minimum_dataSize_;  
}; // class ImuInitialize
};   // namespace Estimator 

#endif  