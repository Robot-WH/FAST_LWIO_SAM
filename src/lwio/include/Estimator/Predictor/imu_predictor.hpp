/*
 * @Copyright(C): Your Company
 * @FileName: 文件名
 * @Author: lwh
 * @Description: 
 * @Others: 
 */

#ifndef _IMU_PREDICTOR_HPP_
#define _IMU_PREDICTOR_HPP_

#include <eigen3/Eigen/Dense>
#include "Sensor/sensor.hpp"
#include "Estimator/state.hpp"
#include "Common/utility.hpp"

namespace Slam3D {

/**
 * @brief 卡尔曼滤波的IMU预测类 
 * @details 变化: 采用的状态(比如是否优化外参)、是否优化重力
 */
class ImuPredictor {
    public:
        /**
         * @brief: 构造函数  IMU运动学模型可设置  
         * @param acc_noise 加速度测量噪声
         * @param gyro_noise 角速度测量噪声
         */            
        ImuPredictor( float const& acc_noise, float const& gyro_noise, float const& acc_bias_noise, 
            float const& gyro_bias_noise) : acc_noise_(acc_noise), gyro_noise_(gyro_noise), 
            acc_bias_noise_(acc_bias_noise), gyro_bias_noise_(gyro_bias_noise) {
            last_imu_ = std::make_shared<ImuData>();
        }
        ~ImuPredictor() {}
        ImuDataConstPtr const& GetLastData() const {
            return last_imu_; 
        }
        
        void SetLastData(ImuDataConstPtr const& data) {
            last_imu_=data;  
        }

        /**
         * @brief: 初始化
         * @param data 初始位置处的IMU数据
         */            
        void Initialize(ImuDataConstPtr const& data) {   
            last_imu_=data;  
        }

        /**
         * @brief 使用IMU的预测环节 
         * @param[out] state 状态, 变化 是否优化外参等等...  
         * @param[in] curr_data 当前时刻的IMU测量  
         * @param[out] cov 预测后的协方差矩阵，变化 是否优化外参、优化重力...
         */ 
        template<typename _StateType, int _StateDim>
        void Predict(_StateType &state, ImuDataConstPtr const& curr_data, 
                                    Eigen::Matrix<double, _StateDim, _StateDim> &cov);
    private:
            
        /**
         * @brief 中值积分预测PVQ
         * @param[in/out] state 上一时刻的状态/推导的下一时刻状态
         * @param curr_imu 当前的imu测量
         * @param acc_bias IMU加速度偏置
         * @param gyro_bias IMU角速度偏置 
         */
        void predictPVQ(StateWithImu &state, ImuDataConstPtr const& curr_imu) {
            assert(state.timestamp_ == last_imu_->timestamp); 
            // 时间间隔
            double const delta_t = curr_imu->timestamp - state.timestamp_;
            double const delta_t2 = delta_t * delta_t;
            // 中值积分  
            // Acc and gyro.
            Eigen::Vector3d const acc_0 = state.Q_ * (last_imu_->acc - state.acc_bias_) + state.G_;
            Eigen::Vector3d const mid_gyro_unbias = 0.5 * (last_imu_->gyro + curr_imu->gyro) - state.gyro_bias_;
            // 角增量向量   陀螺仪测出来的角速度向量就可以等价为旋转向量  
            Eigen::Vector3d const delta_angle_axis = mid_gyro_unbias * delta_t;
            // 更新旋转  
            if (delta_angle_axis.norm() > 1e-12) {
                state.Q_ = state.Q_ * Eigen::Quaterniond(
                        1, delta_angle_axis[0] / 2, delta_angle_axis[1] / 2, delta_angle_axis[2] / 2);
            }
            Eigen::Vector3d const acc_1 = state.Q_ * (curr_imu->acc - state.acc_bias_) + state.G_;
            Eigen::Vector3d const mid_acc_unbias = 0.5 * (acc_0 + acc_1);  
            // nominal state. 
            state.P_ += (state.V_ * delta_t + 0.5 * mid_acc_unbias * delta_t2);
            state.V_ += (mid_acc_unbias * delta_t);
            return;  
        }

        /**
         * @brief: 计算预测过程中协方差矩阵
         * @param {*}
         * @return {*}
         */    
        template<typename _StateType, int _StateDim>
        void updateCovMatrix(_StateType const& state, ImuDataConstPtr const& curr_data, 
                                                        Eigen::Matrix<double, _StateDim, _StateDim> &cov);    
    private:
        // 上一时刻IMU数据  
        ImuDataConstPtr last_imu_; 
        // IMU参数
        float acc_noise_;
        float gyro_noise_;
        float acc_bias_noise_; 
        float gyro_bias_noise_; 
}; // class ImuPredictor

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: 特化  计算不考虑重力优化下的IMU预测协方差矩阵传播   
 * @details StateWithImu 考虑IMU的状态,15 状态的维度 ，不优化重力
 * @param 
 * @return {*}
 */    
template<>
void ImuPredictor::updateCovMatrix<StateWithImu, 15>(StateWithImu const&state, 
    ImuDataConstPtr const& curr_data, Eigen::Matrix<double, 15, 15> &cov) {
    assert(state.timestamp_ == last_imu_->timestamp); 
    // 时间差 
    const double delta_t = curr_data->timestamp - state.timestamp_;
    const double delta_t2 = delta_t * delta_t;
    // Acc and gyro.
    const Eigen::Vector3d acc_unbias = 0.5 * (last_imu_->acc + curr_data->acc) - state.acc_bias_;
    const Eigen::Vector3d gyro_unbias = 0.5 * (last_imu_->gyro + curr_data->gyro) - state.gyro_bias_;
    // Error-state. Not needed.
    // Fx
    Eigen::Matrix<double, 15, 15> Fx = Eigen::Matrix<double, 15, 15>::Identity();
    Fx.block<3, 3>(0, 3)   = Eigen::Matrix3d::Identity() * delta_t;
    Fx.block<3, 3>(3, 6)   = - state.Q_.toRotationMatrix() * 
                                                                GetSkewMatrix(acc_unbias) * delta_t;
    Fx.block<3, 3>(3, 9)   = - state.Q_.toRotationMatrix() * delta_t;
    Eigen::Vector3d delta_angle_axis = gyro_unbias * delta_t;  
    if (delta_angle_axis.norm() > 1e-12) {
        Fx.block<3, 3>(6, 6) = Eigen::AngleAxisd(delta_angle_axis.norm(), 
                                                    delta_angle_axis.normalized()).toRotationMatrix().transpose();
    } else {
        Fx.block<3, 3>(6, 6) = Eigen::Matrix<double, 3, 3>::Identity();
    }
    Fx.block<3, 3>(6, 12)  = - Eigen::Matrix3d::Identity() * delta_t;
    // Fi  IMU噪声转换矩阵   IMU噪声只是影响 速度、旋转、bias 
    Eigen::Matrix<double, 15, 12> Fi = Eigen::Matrix<double, 15, 12>::Zero();
    Fi.block<12, 12>(3, 0) = Eigen::Matrix<double, 12, 12>::Identity();
    // IMU噪声协方差矩阵 
    Eigen::Matrix<double, 12, 12> Qi = Eigen::Matrix<double, 12, 12>::Zero();
    Qi.block<3, 3>(0, 0) = delta_t2 * acc_noise_ * Eigen::Matrix3d::Identity();
    Qi.block<3, 3>(3, 3) = delta_t2 * gyro_noise_ * Eigen::Matrix3d::Identity();
    Qi.block<3, 3>(6, 6) = delta_t * acc_bias_noise_ * Eigen::Matrix3d::Identity();
    Qi.block<3, 3>(9, 9) = delta_t * gyro_bias_noise_ * Eigen::Matrix3d::Identity();
    cov = Fx * cov * Fx.transpose() + Fi * Qi * Fi.transpose();
}

/**
 * @brief: 特化  计算不考虑重力优化下的IMU预测协方差矩阵传播   
 * @param {*}
 * @return {*}
 */    
template<>
void ImuPredictor::Predict<StateWithImu, 15>(StateWithImu &state, 
        ImuDataConstPtr const& curr_data, Eigen::Matrix<double, 15, 15> &cov) {
    assert(state.timestamp_ == last_imu_->timestamp); 
    // 可以采用多种积分方式  RK4、中值
    // IMU运动积分只与状态与IMU测量值有关   因此  采用多态方式进行切换 
    predictPVQ(state, curr_data);   // 状态预测 
    // std::cout<<"PVQ predict - P: "<<state.P_<<" ,V: "
    //             <<state.V_<<std::endl;
    updateCovMatrix(state, curr_data, cov);  // 更新协方差矩阵
    state.timestamp_ = curr_data->timestamp; // 更新状态的时间戳 
    // 保存为上一个imu数据 
    last_imu_ = curr_data; 
}
} // namespace Estimator 

#endif  
