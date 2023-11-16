
#pragma once 
#include <ceres/ceres.h>
#include "../ceres/pose_local_parameterization.hpp"
#include "../preintegration/gyro_preintegration.hpp"

namespace lwio {
namespace estimator {
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief 估计陀螺仪Bias的滑动窗口的估计器
 * 
 */
class GyroBiasSWOEstimate {
public:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    GyroBiasSWOEstimate() : frame_count_{0}, bg_{0.0, 0.0, 0.0}, init_(false) {
        J_ = Eigen::MatrixXd::Zero(3 * Param::WINDOW_SIZE_, 3);  
        e_ = Eigen::VectorXd::Zero(3 * Param::WINDOW_SIZE_, 1);

        for (int i = 0; i < Param::WINDOW_SIZE_; i++) {
                gyro_pre_integrations_[i] = nullptr;
        }
        clearState();  
    } 

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief 对估计器进行初始化
     * @details 设置外参和bias初始值
     * 
     */
    void Initialize(Eigen::Quaterniond imu_q_s, Eigen::Vector3d bg) {
        imu_q_s_ = imu_q_s;
        bg_ = bg;  
        init_ = true; 
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief 只估计陀螺仪的偏置参数 
     *                  融合IMU时 如果只使用IMU的陀螺仪数据 那么可以使用这个估计策略  
     * @param imu_data imu数据   (只使用陀螺仪数据)
     * @param motion 外部传感器的运动观测 (只使用旋转)
     * @param lidarOdom_cov 
     * @return true 
     * @return false 
     */
    bool Estimate(const std::deque<sensor::ImuData>& imu_data, const Eigen::Isometry3d& motion, 
            const Eigen::Matrix<double, 6, 6>& lidarOdom_cov = Eigen::Matrix<double, 6, 6>::Identity()) {
        if (!init_) {
            std::cout << "估计器没有初始化!!" << std::endl;
            return false;  
        }
        // IMU陀螺仪角度预积分
        gyroPreIntegrations(imu_data);
        // 构造最小二乘优化
        // 通过外参 将外部传感器观测的旋转转换到imu系下
        Eigen::Quaterniond q_sensor(motion.linear());   
        Eigen::Quaterniond q_imu_obs(imu_q_s_ * q_sensor * imu_q_s_.inverse());
        q_imu_obs_queue_[frame_count_] = q_imu_obs;  

        for (int i = 0; i < Param::WINDOW_SIZE_; i++) {
            if (gyro_pre_integrations_[i] != nullptr) {
                J_.block<3, 3>(i * 3, 0) = 
                    gyro_pre_integrations_[i]->jacobian_.template block<3, 3>(0, 3);
                e_.block<3, 1>(i * 3, 0) = 
                    // Eigen::Vector3d::Zero();  
                    (gyro_pre_integrations_[i]->delta_q_.inverse() * q_imu_obs_queue_[i]).vec();
            }
        }

        // std::cout << "预积分旋转 delta_q_ w: " << gyro_pre_integrations_[frame_count_]->delta_q_.w() 
        //     << ",vec: " << gyro_pre_integrations_[frame_count_]->delta_q_.vec().transpose() << std::endl;
        // std::cout << "误差: " << (gyro_pre_integrations_[frame_count_]->delta_q_.inverse() * q_imu_obs).vec().transpose() << std::endl; 

        A_ = J_.transpose() * J_;
        b_ = 2 * J_.transpose() * e_;
        // 求解
        Eigen::Vector3d delta_bg = A_.ldlt().solve(b_);
        // 更新状态
        bg_ += delta_bg;  
        std::cout << "在线估计陀螺仪bias,delta_bg: " << delta_bg.transpose() 
            << ",bg_: " << bg_.transpose() << std::endl;
        // 预积分更新
        for (int i = 0; i < Param::WINDOW_SIZE_; i++) {
            if (gyro_pre_integrations_[i] != nullptr) {
                gyro_pre_integrations_[i]->repropagate(bg_);
            }
        }
        // 滑动窗口
        if (frame_count_ == Param::WINDOW_SIZE_ - 1) {
            frame_count_ = 0;
        } else {
            ++frame_count_;
        }
        if (gyro_pre_integrations_[frame_count_] != nullptr) {
            gyro_pre_integrations_[frame_count_]->Reset();  
        }
        return true;  
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    const Eigen::Vector3d& GetBgs() const {
        return bg_;
    }

private:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void clearState() {
        for (int i = 0; i < Param::WINDOW_SIZE_; i++) {
            if (gyro_pre_integrations_[i] != nullptr)
                delete gyro_pre_integrations_[i];
            gyro_pre_integrations_[i] = nullptr;
        }
        bg_.setZero();  
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief 
     * 
     * @param imu_data 
     */
    void gyroPreIntegrations(const std::deque<sensor::ImuData>& imu_data) {
        if (gyro_pre_integrations_[frame_count_] == nullptr) {
            gyro_pre_integrations_[frame_count_] = 
                new GyroPreIntegration(bg_);   // bias用
        }
        for (const auto& imu : imu_data) {
            gyro_pre_integrations_[frame_count_]->push_back(imu.timestamp_, imu.gyro_);
        }
    }

    struct Param {
        static const int WINDOW_SIZE_ = 3;
    };
    bool init_; 
    int frame_count_;     // 当前滑动窗口中，帧的数量  
    // 系统状态 
    GyroPreIntegration *gyro_pre_integrations_[Param::WINDOW_SIZE_];
    Eigen::Quaterniond q_imu_obs_queue_[Param::WINDOW_SIZE_];
    Eigen::Vector3d bg_;     // 陀螺仪bias
    Eigen::Quaterniond imu_q_s_;     // 外参： sensor -> imu 的旋转 
    // 优化信息矩阵
    Eigen::Matrix3d A_;
    Eigen::Vector3d b_;
    Eigen::MatrixXd J_;
    Eigen::VectorXd e_;
};

}
}