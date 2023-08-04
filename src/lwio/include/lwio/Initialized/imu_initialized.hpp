
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
    // 初始化的结果 
    struct Result {
        Eigen::Vector3d gyro_bias_;   
        Eigen::Vector3d acc_bias_; 
        Eigen::Matrix3d R_w_b_;   
    };

    ImuInitialize() : MeanAccLimit_(0.05),  minimum_dataSize_(100) {}
    ImuInitialize(double MeanAccLimit, int minimum_dataSize) 
        : MeanAccLimit_(MeanAccLimit), minimum_dataSize_(minimum_dataSize) {}

    virtual ~ImuInitialize(){}

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief IMU初始化
     * @param[out]  motion 本次imu数据范围内的6dof运动
     * @param[in] imu_buffer imu数据
     */
    bool Initialize(const Eigen::Isometry3d& motion, const std::deque<sensor::ImuData>& imu_buffer) {
        // // 判断IMU数据是否足够   必须要有足够的IMU才能初始化成功
        // std::cout<<" imu_buffer.size(): " << imu_buffer.size() <<" minimum_dataSize_: "
        //     << minimum_dataSize_ <<std::endl;
        // if (imu_buffer.size() < minimum_dataSize_) {
        //     return false;  
        // }
        // // 求均值
        // Eigen::Vector3d sum_acc(0., 0., 0.);
        // Eigen::Vector3d sum_gyro(0., 0., 0.);
        // // 计算均值 
        // for  (auto const& imu_data : imu_buffer) {
        //     sum_acc += imu_data->acc;
        //     sum_gyro += imu_data->gyro;
        // }
        // mean_acc = sum_acc / (double)imu_buffer.size();
        // mean_gyro = sum_gyro / (double)imu_buffer.size();
        // Eigen::Vector3d sum_err2(0., 0., 0.);
        // // 计算加速度标准差
        // for  (const auto imu_data : imu_buffer) {
        //     sum_err2 += (imu_data->acc - mean_acc).cwiseAbs2();   // cwiseAbs2 各个元素平方  
        // }
        // const Eigen::Vector3d sigma_acc = (sum_err2 / ((double)cache_imu.size() - 1)).cwiseSqrt();
        // // 求模长 检查是否运动过于剧烈 
        // if  (sigma_acc.norm() > MeanAccLimit_) {
        //     std::cout << "[CalculateStaticImuMeanData]: Too big sigma_acc: " 
        //     << sigma_acc.transpose() << std::endl;
        //     return false;
        // }

        // // IMU数据中是否有旋转信息 
        // bool rotation_active = false;  
        // // 如果没有旋转信息   采用加速度进行初始化  
        // if ((*imu_buffer.begin())->rot.w() == 0&&
        //     (*imu_buffer.begin())->rot.x() == 0&&
        //     (*imu_buffer.begin())->rot.y() == 0&&
        //     (*imu_buffer.begin())->rot.z() == 0) {
        //     std::cout<<"FilterEstimatorCentreRobot::Initialize ----- no rotation message !" << std::endl;
        // } else {
        //     //rotation_active = true;
        //     std::cout<<"FilterEstimatorCentreRobot::Initialize ----- use rotation message !" << std::endl;
        // }
        // std::vector<Eigen::Vector3d> acc_buf;   // 加速度buf
        // std::vector<Eigen::Vector3d> gyro_buf;  // 陀螺仪buf
        // // 提取数据 
        // for (auto const& imu : imu_buffer) {
        //     acc_buf.push_back(std::move(imu->acc)); 
        //     gyro_buf.push_back(std::move(imu->gyro)); 

        //     if (!rotation_active) {
        //         continue;
        //     }
        // }
        // // 平滑后的等效IMU   用于初始化 
        // ImuDataPtr initialized_imu = std::make_shared<ImuData>(); 
        // initialized_imu->timestamp = timestamp;     // 时间戳和GNSS的相等  

        // // 检查IMU是否可以获取旋转数据  
        // Eigen::Matrix3d R_w_b;  
        // if (!rotation_active) { 
        //     // 靠重力初始化
        //     R_w_b = computeRotationFromGravity(initialized_imu->acc);
        //     BaseType::estimated_state_.common_states_.Q_ = Eigen::Quaterniond(R_w_b).normalized();   
        //     std::cout<<"imu rotation initialize ! R: "<< std::endl << R_w_b << std::endl;
        // } 
        // else { // 直接用旋转初始化 
            
        // }

        // return true;   
    }
    
private:

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief 通过IMU的重力数据恢复旋转  
     * @param Rwi imu->world的旋转  
     * @param smooth_gravity 平滑后的重力 
     * @return 是否计算成功 
     */
    Eigen::Matrix3d const computeRotationFromGravity(Eigen::Vector3d const& smooth_gravity) {
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
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief 陀螺仪偏置  静态初始化 
     * @param gyro_buf 陀螺仪数据 
     */
    Eigen::Vector3d StaticInitializeGyroBias(std::vector<Eigen::Vector3d> const& gyro_buf) {   
        // 就是求平均值
        Eigen::Vector3d sum_value{0, 0, 0};
        for(auto const& gyro : gyro_buf) {
            sum_value += gyro;
        }
        return sum_value / gyro_buf.size();   
    }        
    // 初始化加速度阈值   
    double MeanAccLimit_;  
    // IMU缓存数量阈值
    int minimum_dataSize_;  
}; // class ImuInitialize
};   // namespace Estimator 