
/*
 * @Copyright(C):
 * @Author: lwh
 * @Description: 航迹推测  使用 IMU与odom
 * @Others: 
 */
#pragma once 

#include "Common/utility.hpp"
#include "Common/color.hpp"
#include "Sensor/sensor.hpp"

namespace Slam3D {

/**
 * @brief: 输入imu角速度测量值，里程计速度测量值/位置测量 估计 当前位置
 *                  包含IMU时，可通过IMU数据评估 odom的准确性(是否打滑)
 * 
 */    
class DeadReckon3D {
public:
    // 配置项
    struct Option {
        bool use_odom_motion_ = true;   
    };
    struct Pose {
        double x_ = 0, y_ = 0, z_ = 0;
        double roll_ = 0, pitch_ = 0, yaw_ = 0;  
    };
    DeadReckon3D() {}
    DeadReckon3D(Option option) : option_(option) {
    }
    
    /**
     * @brief: 添加IMU观测
     * @details IMU用于更新角速度以及计算角度变化量, 角度变化量用于去除畸变 
     * @param acc_bias 加速度bias 
     * @param gyro_bias 角速度bias 
     */            
    void AddImuData(ImuData const& imu, Eigen::Vector3d const& acc_bias, 
            Eigen::Vector3d const& gyro_bias) {
        if (recorded_imu_rot_queue_.empty()) {
            last_gyro_ = imu.gyro_ - gyro_bias; 
            recorded_imu_rot_queue_.emplace_back(imu.timestamp_, 0, 0, 0);   // 初始imu 旋转设为0   
            // 保存imu数据   用于 融合odom的3D里程计解算 
            cache_imu_for_odom_.emplace_back(imu.timestamp_, imu.acc_ - acc_bias,  // 去除bias 
                last_gyro_, imu.rot_);
            return;
        }
        ImuRot& last_rot = recorded_imu_rot_queue_.back();   // 上一时刻的旋转
        double dt = imu.timestamp_ - last_rot.timestamp_;
        double curr_roll_rot = last_rot.roll_rot_ + dt * 0.5 * (last_gyro_[0] + (imu.gyro_[0] - gyro_bias[0]));
        double curr_pitch_rot = last_rot.pitch_rot_ + dt * 0.5 * (last_gyro_[1] + (imu.gyro_[1] - gyro_bias[1]));
        double curr_yaw_rot = last_rot.yaw_rot_ + dt * 0.5 * (last_gyro_[2] + (imu.gyro_[2] - gyro_bias[2]));
        Utility::NormalizeAngle(curr_roll_rot);
        Utility::NormalizeAngle(curr_pitch_rot);
        Utility::NormalizeAngle(curr_yaw_rot);

        recorded_imu_rot_queue_.emplace_back(imu.timestamp_, curr_roll_rot, curr_pitch_rot, curr_yaw_rot); 
        // 
        last_gyro_ =  imu.gyro_ - gyro_bias; 
        cache_imu_for_odom_.emplace_back(imu.timestamp_, imu.acc_ - acc_bias, last_gyro_, imu.rot_);
        // 保留 1000ms 以内的数据
        while (recorded_imu_rot_queue_.back().timestamp_ - 
                recorded_imu_rot_queue_.front().timestamp_ > 1000) {
            recorded_imu_rot_queue_.pop_front(); 
        }
        while (cache_imu_for_odom_.back().timestamp_ - 
                cache_imu_for_odom_.front().timestamp_ > 1000) {
            cache_imu_for_odom_.pop_front(); 
        }
    }

    /**
     * @brief: 添加里程计观测，进行3D航迹推算
     * @details 该里程计数据仅有车体方向的速度以及车体yaw旋转角速度信息
     *                     1、若有imu角速度观测，则是用 imu角速度 + odom车体速度观测 进行航迹推算
     *                      2、若没有imu角速度，则使用odom角速度以及车体速度观测，在上一时刻pose的基础下递推
     */            
    void AddOdometryData(OdomData const& curr_odom) {
        cache_odom_.push_back(curr_odom); 
        if (cache_odom_.size() < 2) {
            return;
        }
        OdomData& odom = cache_odom_.front();  
        cache_odom_.pop_front(); 
        
        if (last_motion_.timestamp_ == -1) {
            // 直接使用原始的odom数据初始化运动信息 
            last_motion_.timestamp_ = odom.timestamp_;
            last_motion_.linear_velocity_.x() = odom.velocity_;   // 车辆运动学约束   只有x方向的速度
            last_motion_.linear_velocity_.y() = 0;
            last_motion_.linear_velocity_.z() = 0;
            last_motion_.angle_velocity_.x() = 0;   // roll
            last_motion_.angle_velocity_.y() = 0; // pitch
            last_motion_.angle_velocity_.z() = odom.yaw_angular_vel_; // yaw
            last_odom_ = odom; 
            return;
        }
        
        bool use_gyro_ = false;   
        curr_motion_.timestamp_ = odom.timestamp_;
        // 若有IMU的角速度观测，则使用IMU角速度观测 代替odom角速度
        if (!cache_imu_for_odom_.empty()) {
            // 插值出本次 odom 时间戳处 旋转角速度
            if (cache_imu_for_odom_.back().timestamp_ < odom.timestamp_) {
                cache_imu_for_odom_.clear();  
            } else {
                ImuData before_imu;       // odom之前的一个imu数据
                while (cache_imu_for_odom_.front().timestamp_ <= odom.timestamp_) {
                    before_imu = cache_imu_for_odom_.front();  
                    cache_imu_for_odom_.pop_front(); 
                }
                if (before_imu.timestamp_ <= odom.timestamp_ && before_imu.timestamp_ > 0) {
                    ImuData& after_imu = cache_imu_for_odom_.front();   // odom后一个imu数据
                    // 进行插值 
                    curr_motion_.angle_velocity_ = Utility::Interpolate<Eigen::Vector3d>(before_imu.gyro_, 
                        after_imu.gyro_, before_imu.timestamp_, after_imu.timestamp_, curr_motion_.timestamp_); 
                    use_gyro_ = true;  
                }
            }
        }
        
        // if (use_gyro_) 
        //     std::cout<<common::GREEN<<"use_gyro_"<<common::RESET<<std::endl;
        // else
        //     std::cout<<common::RED<<"no use_gyro_"<<common::RESET<<std::endl;

        double delta_time = curr_motion_.timestamp_ - last_motion_.timestamp_; 
        Eigen::Vector3d mid_linear_velocity, mid_angular_velocity;   // 中值速度  
        // 不使用原始odom的速度信息  则通过位姿差计算速度，角速度优先使用IMU的
        if (!option_.use_odom_motion_) {
            Eigen::Vector2d linear_v;  
            linear_v = (odom.pos_xy_ - last_odom_.pos_xy_) / delta_time;
            curr_motion_.linear_velocity_.x() = linear_v.norm();    // 车体方向的速度 
            mid_linear_velocity = curr_motion_.linear_velocity_;  
            if (!use_gyro_) {
                /**
                 * @todo: 使用odom的运动模型确定yaw角速度, 这里暂时直接用odom的角速度输出代替
                 */
                curr_motion_.angle_velocity_.z() = odom.yaw_angular_vel_;   
            } 
            mid_angular_velocity = (curr_motion_.angle_velocity_ + last_motion_.angle_velocity_) / 2;  
            last_odom_ = odom; 
        } else {
            curr_motion_.linear_velocity_.x() = odom.velocity_;    // 当前运动的速度 直接 等于 odom的测量
            // 使用中值法
            mid_linear_velocity = (curr_motion_.linear_velocity_ + last_motion_.linear_velocity_) / 2;  
            if (!use_gyro_) {
                curr_motion_.angle_velocity_.z() = odom.yaw_angular_vel_;   
            }
            mid_angular_velocity = (curr_motion_.angle_velocity_ + last_motion_.angle_velocity_) / 2;  
        }
        Pose curr_pose; 
        // 差分运动学解算  
        curr_pose.roll_ = last_pose_.roll_ + delta_time * mid_angular_velocity.x();
        curr_pose.pitch_ = last_pose_.pitch_ + delta_time * mid_angular_velocity.y();
        curr_pose.yaw_ = last_pose_.yaw_ + delta_time * mid_angular_velocity.z();
        curr_pose.x_ = last_pose_.x_ + mid_linear_velocity.x() * delta_time * 
            cos(delta_time * mid_angular_velocity.y() / 2 + last_pose_.pitch_) * 
            cos(delta_time * mid_angular_velocity.z() / 2 + last_pose_.yaw_);
        curr_pose.y_ = last_pose_.y_ + mid_linear_velocity.x() * delta_time * 
            cos(delta_time * mid_angular_velocity.y() / 2 + last_pose_.pitch_) * 
            sin(delta_time * mid_angular_velocity.z() / 2 + last_pose_.yaw_);
        curr_pose.z_ = last_pose_.z_ + 
            mid_linear_velocity.x() * delta_time * sin(delta_time * mid_angular_velocity.y() / 2 + last_pose_.pitch_); 
        last_pose_ = curr_pose;
        recorded_odom_pose_queue_.emplace_back(odom.timestamp_, last_pose_); 
        // 更新当前运动信息    
        last_motion_ = curr_motion_;
    }

    // /**
    //  * @brief: 添加位姿观测 
    //  * @details: 匀速运动学模型计算 线速度和角速度 
    //  *                      1、没有IMU以及odom时，采用这里计算出来的匀速运动学模型进行预测
    //  *                      2、有IMU无odom，使用匀速运动模型的线速度，以及IMU的角速度
    //  *                      3、有IMU与odom，不使用匀速运动学模型
    //  */            
    // void AddPoseData() {
        
    // }

    // /**
    //  * @brief: 推测时间time时的车体位姿
    //  * @details: 
    //  * @param undefined
    //  * @return {*}
    //  */        
    // bool ExtrapolatorPose(double const& time, Pose &res) {
    //     // // 有imu的数据   则使用 Imu的数据  插值出旋转 
    //     // // if (recorded_imu_rot_queue_.empty()) {

    //     // // }
    //     // // 没有odom数据 ，则使用imu预测旋转，或匀速运动学模型预测
    //     // if (recorded_odom_pose_queue_.empty()) {
    //     //     // 如果时间晚于 上一次pose的时间则 出错
    //     // }
    //     // if (time < recorded_odom_pose_queue_.front().first) {
    //     //     std::cout<<"time < recorded_odom_pose_queue_.front().first"<<std::endl;
    //     //     return false; 
    //     // }
    //     // // 如果时间戳 在odom 数据覆盖范围之外  则进行预测  否则就是插值  
    //     // if (time > recorded_odom_pose_queue_.back().first) {
    //     // }
    //     // //std::cout<<"time: "<<time<<std::endl;
    //     // // 插值 
    //     // Pose2d before_pose, after_pose; 
    //     // double before_time, after_time; 
    //     // // 使用odom的pose 
    //     // while(!recorded_odom_pose_queue_.empty()) {
    //     //     if (recorded_odom_pose_queue_.front().first >= time) {
    //     //         after_pose = recorded_odom_pose_queue_.front().second;
    //     //         after_time = recorded_odom_pose_queue_.front().first;
    //     //         recorded_odom_pose_queue_.emplace_front(before_time, before_pose); 
    //     //         //std::cout<<"after_time: "<<after_time<<std::endl;
    //     //         break; 
    //     //     } else {
    //     //         before_pose = recorded_odom_pose_queue_.front().second;
    //     //         before_time = recorded_odom_pose_queue_.front().first;
    //     //         //std::cout<<"before_time: "<<before_time<<std::endl;
    //     //     }
    //     //     recorded_odom_pose_queue_.pop_front();  
    //     // }
    //     // res = Utility::Interpolate<Pose2d>(before_pose, after_pose, before_time, after_time, time); 
    //     // return true;  
    // }

    /**
     * @brief: 获取航迹推算的当前位姿
     */        
    Pose GetDeadReckoningPose() const {
        return last_pose_; 
    }

private:
    // 运动信息   局部坐标系下的 线速度 + 角速度
    struct MotionInfo {
        double timestamp_ = -1; 
        Eigen::Vector3d linear_velocity_{0., 0., 0.};   // x，y, z方向   是相对于 车体坐标系的
        Eigen::Vector3d angle_velocity_{0., 0., 0.}; // roll, pitch, yaw   相对于车体坐标系 
    };
    struct ImuRot {
        ImuRot(double const& timestamp, double const& roll_rot, 
            double const& pitch_rot, double const& yaw_rot) : timestamp_(timestamp), yaw_rot_(yaw_rot),
            roll_rot_(roll_rot), pitch_rot_(pitch_rot) {}
        double timestamp_ = -1; 
        double roll_rot_;
        double pitch_rot_;
        double yaw_rot_;
    };
    Option option_; 
    Eigen::Vector3d last_gyro_;  // 上一时刻IMU的角速度
    OdomData last_odom_;  
    MotionInfo last_motion_, curr_motion_;  
    Pose last_pose_; 
    std::deque<OdomData> cache_odom_;   // odom的缓存 
    std::deque<ImuData> cache_imu_for_odom_;  
    std::deque<std::pair<double, Pose>> recorded_odom_pose_queue_;   // 记录里程计解算的位姿   用于激光去畸变以及匹配预测
    std::deque<ImuRot> recorded_imu_rot_queue_; 
    Eigen::Matrix3d rot_base_imu_ = Eigen::Matrix3d::Identity();   // imu到base的旋转外参 
}; // class
} // namespace 