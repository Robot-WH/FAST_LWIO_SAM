
#pragma once 

#include <eigen3/Eigen/Dense>
#include "../Sensor/gnss_data_process.hpp"
#include "../Sensor/sensor.hpp"

namespace lwio {

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: IMU 静止初始化 
 * @details: 
 */    
class ImuInitialize {
public:
    ImuInitialize() {
    }
    /**
     * @brief: 支持静止初始化以及运动初始化
     * @param in data 
     * @param out gyro_bias 
     * @param out acc_bias
     * @param out R_gb
     * @return {*}
     */        
    bool Initialize(const std::deque<sensor::ImuData>& data, Eigen::Vector3d& gyro_bias,
                                    Eigen::Vector3d& acc_bias, Eigen::Matrix3d& R_gb) {
        // 静止初始化
        std::cout << SlamLib::color::YELLOW << "imu static init ...... " << SlamLib::color::RESET << std::endl;

        if (staticInit(data, gyro_bias, R_gb)) {
            std::cout << SlamLib::color::GREEN << "imu static init success...... " 
                << SlamLib::color::RESET << std::endl;
            return true;  
        }
        return false;  
    }

    /**
     * @brief 初始化后进行的在线标定
     *                  连续若干帧检测到处与静止状态，则求解陀螺仪的Bias
     * 
     * @param gyro 
     */
    void OnlineCalibGyroBias(const Eigen::Vector3d& gyro, Eigen::Vector3d& gyro_bias) {
        if (std::fabs(gyro[0]) < 0.01 && std::fabs(gyro[1]) < 0.01) {
            avg_gyro_[0] += (gyro[0] - avg_gyro_[0]) / static_init_N_;
            avg_gyro_[1] += (gyro[1] - avg_gyro_[1]) / static_init_N_;
            ++static_init_N_;
            if (static_init_N_ > 100) {
                gyro_bias[0] += avg_gyro_[0];  
                gyro_bias[1] += avg_gyro_[1];  
            } else {
                return;  
            }
        } 
        static_init_N_ = 1; 
        avg_gyro_[0] = 0.0;
        avg_gyro_[1] = 0.0;
    }

private:
    /**
     * @brief: 静止初始化
     * @details: 积累连续的20个数据即可进行初始化
     * @return 是否成功
     */        
    bool staticInit(const std::deque<sensor::ImuData>& packet, Eigen::Vector3d& gyro_bias,
                                    Eigen::Matrix3d& R_gb) {

        for (const auto& data : packet) {
            if (static_init_N_ > 1) {
                // 估计样本方差   
                // 陀螺仪方差
                var_gyro_[0] = var_gyro_[0] * (static_init_N_ - 2) / (static_init_N_ - 1)
                    + pow((data.gyro_[0] - avg_gyro_[0]), 2) / static_init_N_;  
                var_gyro_[1] = var_gyro_[1] * (static_init_N_ - 2) / (static_init_N_ - 1)
                    + pow((data.gyro_[1] - avg_gyro_[1]), 2) / static_init_N_;  
                var_gyro_[2] = var_gyro_[2] * (static_init_N_ - 2) / (static_init_N_ - 1)
                    + pow((data.gyro_[2] - avg_gyro_[2]), 2) / static_init_N_;  
                // 加速度方差 
                var_acc_[0] = var_acc_[0] * (static_init_N_ - 2) / (static_init_N_ - 1)
                    + pow((data.acc_[0] - avg_acc_[0]), 2) / static_init_N_;  
                var_acc_[1] = var_acc_[1] * (static_init_N_ - 2) / (static_init_N_ - 1)
                    + pow((data.acc_[1] - avg_acc_[1]), 2) / static_init_N_;  
                var_acc_[2] = var_acc_[2] * (static_init_N_ - 2) / (static_init_N_ - 1)
                    + pow((data.acc_[2] - avg_acc_[2]), 2) / static_init_N_;  
            }
            // 均值   
            avg_gyro_ += (data.gyro_ - avg_gyro_) / static_init_N_;
            avg_acc_ += (data.acc_ - avg_acc_) / static_init_N_; 
            ++static_init_N_;
        }

        if (static_init_N_ > 20) {
            std::cout << SlamLib::color::YELLOW << "avg_gyro_: " << avg_gyro_.transpose()
                << ",var_gyro_: " << var_gyro_.transpose() << SlamLib::color::RESET << std::endl; 
            std::cout << SlamLib::color::YELLOW << "avg_acc_: " << avg_acc_.transpose()
                << ",var_acc_: " << var_acc_.transpose() << SlamLib::color::RESET << std::endl;
            // 如果方差和均值足够小，则认为该均值估计可信
            if (std::fabs(var_acc_[0]) < 1e-4 && std::fabs(var_gyro_[0]) < 1e-7 &&
                    std::fabs(var_acc_[1]) < 1e-4 && std::fabs(var_gyro_[1]) < 1e-7 &&
                    std::fabs(var_acc_[2]) < 1e-4 && std::fabs(var_gyro_[2]) < 1e-7) {
                gyro_bias = avg_gyro_;  
                R_gb = getRotationFromGravity(avg_acc_);  
                // 为了在线估计陀螺仪bias的任务，这里需要对下面数据进行初始化
                static_init_N_ = 1; 
                avg_gyro_ = {0.0, 0.0, 0.0};
                return true; 
            } else {
                // 重新估计 
                static_init_N_ = 1; 
                avg_gyro_ = {0.0, 0.0, 0.0};
                var_gyro_ = {0.0, 0.0, 0.0};
                avg_acc_ = {0.0, 0.0, 0.0};
                var_acc_ = {0.0, 0.0, 0.0};
            }
        }

        return false; 
    }

    /**
     * @brief Get the Rotation From Gravity object
     * 
     * @param gravity 重力向量指向z轴正方向 (0,0,9.8)   
     * @return Eigen::Matrix3d 
     */
    Eigen::Matrix3d getRotationFromGravity(const Eigen::Vector3d& gravity) {
        const Eigen::Vector3d z_axis = gravity.normalized();   // 导航系z轴在载体系下的表示
        const Eigen::Vector3d x_axis =  // 导航系x轴在载体系下的表示
            (Eigen::Vector3d::UnitX() - z_axis * z_axis.transpose() * Eigen::Vector3d::UnitX()).normalized();
        // 导航系y轴在载体系下的表示    
        const Eigen::Vector3d y_axis = (z_axis.cross(x_axis)).normalized();
        // 知道G系各轴在I系下的投影，将G系各轴在I系下的投影向量依次摆放在旋转矩阵的列向量上，
        // 则该旋转矩阵可将G系下的向量变换到I系
        Eigen::Matrix3d I_R_G;
        I_R_G.block<3, 1>(0, 0) = x_axis;
        I_R_G.block<3, 1>(0, 1) = y_axis;
        I_R_G.block<3, 1>(0, 2) = z_axis;

        return I_R_G.transpose();
    }

    bool motionInit() {
        return false; 
    }
    
    uint16_t static_init_N_ = 1; 
    float roll_avg_ = 0;
    float pitch_avg_ = 0;
    Eigen::Vector3d avg_acc_{0.0, 0.0, 0.0};
    Eigen::Vector3d var_acc_{0.0, 0.0, 0.0};
    Eigen::Vector3d var_gyro_{0.0, 0.0, 0.0};
    Eigen::Vector3d avg_gyro_{0.0, 0.0, 0.0};
}; // class ImuInitialize
};   // namespace Estimator 