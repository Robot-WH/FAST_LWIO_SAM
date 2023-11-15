
#pragma once 
#include <iomanip> 
#include "glog/logging.h"
#include "SlamLib/tic_toc.hpp"
#include "SlamLib/Common/point_type.h"
#include "lwio/system.h"
#include "Estimator/SWO/SlidingWindowOptimizeEstimator.hpp"

namespace lwio {
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename _PointT>
System<_PointT>::System(std::string param_path) : param_path_(param_path) {
    std::cout << "构造System..." << std::endl;
    YAML::Node yaml = YAML::LoadFile(param_path);
    // 读取外参  
    std::vector<float> data =    yaml["extrinsic"]["lidar_to_imu"]["rot"].as<std::vector<float>>(); 
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            imu_R_lidar_(row, col) = data[3 * row + col];
        }
    }
    data = yaml["extrinsic"]["lidar_to_imu"]["trans"].as<std::vector<float>>();
    for (int i = 0; i < 3; ++i) {
        imu_t_lidar_[i] = data[i]; 
    }

    std::cout << "lidar_to_imu extrinsic rot: " << std::endl 
        << imu_R_lidar_.matrix() << std::endl
        << "trans: " << imu_t_lidar_.transpose() << std::endl;
    // 是否开启初始化标定
    bool calib_init = yaml["extrinsic"]["calib_init"].as<bool>();
    if (calib_init) {
        EXTRINSIC_SET_ = false;
    } else {
        EXTRINSIC_SET_ = true;  
    }
    // imu的使用方式，只用陀螺仪数据或使用全部的数据
    int imu_use_type = yaml["imu_use_type"].as<int>();
    if (!imu_use_type) {
        imu_use_type_ = ImuUseType::only_gyro;
        std::cout << "imu_use_type_: " << "仅仅使用IMU的陀螺仪数据" << std::endl;
    } else {
        imu_use_type_ = ImuUseType::full;
        std::cout << "imu_use_type_: " << "使用IMU的全部数据" << std::endl;
    }
    // 运动方式  - 平面运动、空间任意运动
    PLANE_MOTION_ = yaml["plane_motion"].as<bool>();
    preprocess_.reset(new LidarPreProcess<_PointT>(param_path_));    
    // segmentation_.reset(new PointCloudSegmentation<_PointT>(param_path_)); 
    typename DirectGroundDetect<_PointT>::Option ground_detect_option;
    ground_detect_.reset(new DirectGroundDetect<_PointT>(ground_detect_option));  
    // 构造tracker  
    lidar_trackers_.reset(new LidarTracker<_PointT>(param_path_)); 
    pose_ = Eigen::Isometry3d::Identity(); 
    // 启动估计线程
    process_thread_ = std::thread(&System::processMeasurements, this); 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: 对于Lidar数据的处理
 * @param data
 * @return {*}
 */                
template<typename _PointT>
void System<_PointT>::InputData(LidarData<_PointT>& curr_data) {        
    SlamLib::time::TicToc tt; 
    // 直接法 预处理
    SlamLib::PCLPtr<_PointT> filtered_points = preprocess_->Process(curr_data); 
    tt.toc("降采样：");
    // 地面分割 + 聚类
    SlamLib::PCLPtr<_PointT> unground_points(new pcl::PointCloud<_PointT>);
    SlamLib::PCLPtr<_PointT> ground_points(new pcl::PointCloud<_PointT>);
    // segmentation_->Process(curr_data.pointcloud_ptr_, unstable_points, stable_points, ground_points);  
    std::vector<uint32_t> ground_index; 
    ground_detect_->GroundDetect(*filtered_points, ground_index, *ground_points);  // 地面检测 
    tt.toc("降采样 + 地面分割：");

    SlamLib::CloudContainer<_PointT> feature_points;  
    feature_points.ori_points_num = curr_data.pointcloud_ptr_->size();  
    feature_points.timestamp_start_ = curr_data.timestamp_; 
    feature_points.timestamp_end_ = curr_data.end_timestamp_; 
    feature_points.feature_data_.insert(std::make_pair("filtered", filtered_points));  // 用于匹配
    feature_points.pointcloud_data_.insert(std::make_pair("ground_points", ground_points));  // 地面点  
    // 处理好的点云数据 放置到缓存队列中  等待  估计器使用  
    lidar_sm_.lock();
    processed_lidar_buf_.emplace_back(std::move(feature_points));  
    std::cout << "processed_lidar_buf_ size: " << processed_lidar_buf_.size() << std::endl;
    lidar_sm_.unlock();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: 对于IMU数据的处理
 */    
template<typename _PointT>
void System<_PointT>::InputData(const sensor::ImuData& data) {
    imu_sm_.lock();
    // std::cout << "IMU信息" << std::endl;
    imu_buf_.push_back(data);  
    imu_sm_.unlock(); 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: 对于GNSS数据的处理 
 */            
template<typename _PointT>    
void System<_PointT>::InputData(sensor::GnssData& data) {
    gnss_sm_.lock();
    gnss_buf_.push_back(std::move(data));
    gnss_sm_.unlock(); 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: 对于轮速数据的处理 
 */            
template<typename _PointT>    
void System<_PointT>::InputData(sensor::OdomData& data) {
    odom_sm_.lock();
    // std::cout << "轮速信息" << std::endl;
    odom_buf_.push_back(std::move(data));
    odom_sm_.unlock(); 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief  使用odom和imu的航迹推算
 *                  1、使用IMU的角速度测量以及轮速odom的线速度测量
 *                  2、IMU的X轴与轮速的正方向平行
 *                  3、odom的频率低于imu, odom和imu首尾两个数据的时间戳相同   
*/
template<typename _PointT>    
Eigen::Isometry3d System<_PointT>::deadReckoning(std::deque<sensor::OdomData> const& wheelOdom_container, 
                                            std::deque<sensor::ImuData> const& imu_container) {
    CHECK_EQ(wheelOdom_container.front().timestamp_, imu_container.front().timestamp_);
    CHECK_EQ(wheelOdom_container.back().timestamp_, imu_container.back().timestamp_);

    Eigen::Vector3d linear_velocity{0.0, 0.0, 0.0};   // x，y, z方向   是相对于 车体坐标系的
    Eigen::Vector3d last_trans{0., 0., 0.}; 

    Eigen::Matrix3d R_o_i = Eigen::Matrix3d::Identity();  

    sensor::OdomData back_odom = wheelOdom_container[0];  
    double last_time = imu_container[0].timestamp_; 
    double delta_time = 0; 
    int curr_odom_point = 0;

    for (int i = 1; i < imu_container.size(); ++i) {
        
        if (back_odom.timestamp_ < imu_container[i].timestamp_) {
            // 更新线速度信息
            linear_velocity.x() = back_odom.velocity_;
            back_odom = wheelOdom_container[++curr_odom_point];
        }

        /////////////////////////差分运动学解算  
        delta_time = imu_container[i].timestamp_ - last_time;  
        last_time = imu_container[i].timestamp_;
        // 更新姿态
        const Eigen::Vector3d unbiased_gyro = imu_container[i].gyro_ - sensor_param_.imu_.gyro_bias_;   // 去偏置 
        const Eigen::Vector3d angle_vec = unbiased_gyro * delta_time;
        const double angle = angle_vec.norm();
        const Eigen::Vector3d axis = angle_vec / angle;
        Eigen::Matrix3d delta_rot = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
        R_o_i = R_o_i * delta_rot;    // 旋转矩阵姿态更新  
        Eigen::Vector3d curr_trans = last_trans + R_o_i * linear_velocity * delta_time;
        // 更新当前运动信息    
        last_trans = curr_trans;
    }
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity(); 
    T.linear() = R_o_i;
    T.translation() = last_trans;
    return T;  
}

/**
 * @brief 没有轮速计只有IMU时的航迹推算  
 *                  此时采用匀速运动模型  
 * @tparam _PointT 
 * @param velocity 匀速运动学模型 - 使用上一时刻的速度信息
 * @param imu_container 
 * @return Eigen::Isometry3d 
 */
template<typename _PointT>    
Eigen::Isometry3d System<_PointT>::deadReckoning(Eigen::Vector3d const& velocity, 
                                            std::deque<sensor::ImuData> const& imu_container) {
    Eigen::Vector3d last_trans{0., 0., 0.}; 
    Eigen::Matrix3d R_o_i = Eigen::Matrix3d::Identity();     // 第 i帧到origin的旋转 

    double last_time = imu_container[0].timestamp_; 
    double delta_time = 0; 

    for (int i = 1; i < imu_container.size(); ++i) {
        /////////////////////////差分运动学解算  
        delta_time = imu_container[i].timestamp_ - last_time;  
        last_time = imu_container[i].timestamp_;
        // 更新姿态
        const Eigen::Vector3d unbiased_gyro = imu_container[i].gyro_ - sensor_param_.imu_.gyro_bias_;   // 去偏置 
        const Eigen::Vector3d angle_vec = unbiased_gyro * delta_time;
        const double angle = angle_vec.norm();
        const Eigen::Vector3d axis = angle_vec / angle;
        Eigen::Matrix3d delta_rot = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
        R_o_i = R_o_i * delta_rot;    // 旋转矩阵姿态更新  
        last_trans += delta_time * velocity; 
    }
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity(); 
    // T.linear() = sensor_param_.imu_.lidar_R_imu_ * R_o_i * sensor_param_.imu_.lidar_R_imu_.inverse();
    T.linear() = R_o_i;
    // Eigen::Quaterniond rot_0 = imu_container.front().rot_;  
    // Eigen::Quaterniond rot_1 = imu_container.back().rot_;  
    // T.linear() = sensor_param_.imu_.lidar_R_imu_.inverse() * (rot_1.inverse() * rot_0).toRotationMatrix()
    //                         * sensor_param_.imu_.lidar_R_imu_; 
    T.translation() = last_trans;
    return T;  
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief 融合估计器处理线程，基本逻辑如下：
 *                  1、没有激光数据时，执行INS定位
 *                  2、有激光，没有GNSS，执行LIO
 *                  3、有激光，有GNSS
 */
template<typename _PointT>    
void System<_PointT>::processMeasurements() {
    uint16_t wait_time = 0;     // 用于检测传感器离线
    std::deque<sensor::ImuData> imu_selected;  
    std::deque<sensor::OdomData> wheel_odom_selected; 

    while (1) {
        std::shared_lock<std::shared_mutex> s_l(lidar_sm_);   // 读锁
        // 如果有待处理的激光数据 
        if (processed_lidar_buf_.size()) {
            auto& curr_lidar_data = processed_lidar_buf_.front();  
            s_l.unlock(); 
            ++wait_time; 
            SlamLib::time::TicToc tt;
            // 提取出包围lidar的imu数据, 
            if (syncSensorData(curr_lidar_data.timestamp_start_, curr_lidar_data.timestamp_end_, 
                    wheel_odom_selected, imu_selected) || wait_time > 50) {
                // 自动检测传感器是否在线 
                // 当 wait_time > 50 认为有传感器离线了
                if (wait_time > 50) {
                    if (!imu_buf_.empty() &&
                            imu_buf_.back().timestamp_ < curr_lidar_data.timestamp_end_) {
                        imu_buf_.clear();  
                        std::cout << SlamLib::color::RED << "---------------------imu data loss !-----------------------" 
                            << SlamLib::color::RESET << std::endl; 
                    }
                }

                wait_time = 0; 
                tt.toc("syncSensorData ");

                Eigen::Isometry3d DR_motion = Eigen::Isometry3d::Identity(); 
                
                // 如果系统初始化了，那么进行dead - reckoning 以及激光去畸变
                // dead - reckoning 
                if (!imu_selected.empty()) {
                    // 有IMU 
                    if (IMU_INIT_) {
                        if (!wheel_odom_selected.empty()) {
                            // 轮速-IMU 组合
                            std::cout << SlamLib::color::GREEN << "--------------------- imu - wheel deadReckoning !-----------------------" 
                                << SlamLib::color::RESET << std::endl; 
                            // std::cout << "wheel_odom_selected size: " << wheel_odom_selected.size() << std::endl;
                            // std::cout << "imu_selected size: " << imu_selected.size() << std::endl;
                            // std::cout << std::setprecision(15) << "odom first time : " << wheel_odom_selected.begin()->timestamp_
                            //  << ", lidar start time: " << curr_lidar_data.timestamp_start_ << ", odom end time: "
                            //  << wheel_odom_selected.back().timestamp_ << ", lidar end time:"
                            //  << curr_lidar_data.timestamp_end_ << std::endl;
                            DR_motion = deadReckoning(wheel_odom_selected, imu_selected);
                            mode_ = Mode::lwio;  
                        } else {
                            // 纯IMU
                            std::cout << SlamLib::color::GREEN << "--------------------- imu deadReckoning !-----------------------" 
                                << SlamLib::color::RESET << std::endl; 
                            DR_motion = deadReckoning(filtered_velocity_, imu_selected);
                            mode_ = Mode::lio;  
                        }
                    } else {
                        std::cout << SlamLib::color::YELLOW << "--------------------- lidar odometry !-----------------------" 
                            << SlamLib::color::RESET << std::endl; 
                    }
                } else {
                    // 纯轮速  
                    if (!wheel_odom_selected.empty()) {
                        // 轮速
                        std::cout << SlamLib::color::GREEN << "--------------------- pure wheel deadReckoning!-----------------------" 
                            << SlamLib::color::RESET << std::endl; 
                        mode_ = Mode::lwo;  
                    } else {
                        std::cout << SlamLib::color::YELLOW << "--------------------- lidar odometry !-----------------------" 
                            << SlamLib::color::RESET << std::endl; 
                    }
                }

                Eigen::Isometry3d lidar_motion = Eigen::Isometry3d::Identity(); 
                // 如果 imu 与 lidar的外参知道，那么根据外参预测lidar的运动
                if (EXTRINSIC_SET_) {
                    lidar_motion.linear() = imu_R_lidar_.inverse() * DR_motion.linear() * imu_R_lidar_;
                    lidar_motion.translation() = imu_R_lidar_.inverse() * 
                        (DR_motion.linear() * imu_t_lidar_ + DR_motion.translation() - imu_t_lidar_);
                }
                // 去畸变 
                
                // 激光里程计 
                // TicToc tt;
                std::cout << "predict lidar_motion: " << std::endl 
                    << lidar_motion.matrix() << std::endl;
                static Eigen::Isometry3d predict_pose = Eigen::Isometry3d::Identity(); 
                predict_pose = predict_pose * lidar_motion; 
                lidar_trackers_->Solve(curr_lidar_data, lidar_motion);   
                std::cout << "matched motion: " << std::endl 
                    << lidar_motion.matrix() << std::endl;
                tt.toc("lidar tracker "); 
                pose_ = pose_ * lidar_motion; // 当前帧的绝对运动   
                if (EXTRINSIC_SET_) {
                    estimateBodyLinearVelocity(lidar_motion, curr_lidar_data.timestamp_end_ - curr_lidar_data.timestamp_start_);
                }
                // 融合状态估计器 - 可选择 ESKF、非线性优化  
                if (gyro_bias_Estimate_.Estimate(imu_selected, lidar_motion)) {
                    sensor_param_.imu_.gyro_bias_ = gyro_bias_Estimate_.GetBgs();  
                }
                // 如果没有初始化，那么要进行初始化操作
                // 初始化： 1、初始化IMU内参  2、初始化 IMU/IMU&Odom-Lidar的外参   3、初始化估计器    
                if (!SYSTEM_INIT_) {
                    // IMU_INIT_ = true;  
                    // IMU 静止初始化
                    if (!IMU_INIT_ && imu_init_.Initialize(imu_selected, sensor_param_.imu_.gyro_bias_, 
                            sensor_param_.imu_.acc_bias_, state_.R_gb_)) {
                        IMU_INIT_ = true;  
                    }
                    // 如果IMU标定完成，那么可以进行外参标定
                    if (IMU_INIT_) {
                        // 如果不使用设定的外参，那么要进行外参标定
                        if (!EXTRINSIC_SET_) {
                            // 平面运动
                            if (PLANE_MOTION_) {
                                if (imu_use_type_ == ImuUseType::only_gyro && mode_ == Mode::lwio) {
                                    if (extrinsics_init_.AddPose(DR_motion, lidar_motion)) {
                                        // 先标定旋转 然后标定平移 
                                        if (extrinsics_init_.CalibExRotationPlanar()) {
                                            if (extrinsics_init_.CalibExTranslationPlanar()) {
                                            }
                                        }
                                    }
                                } else {
                                    std::cout << SlamLib::color::RED << 
                                        "当前处与平面运动，但是数据丰富度不足，无法完成外参标定!"
                                        << SlamLib::color::RESET << std::endl; 
                                }
                            } else {
                                if (imu_use_type_ == ImuUseType::only_gyro && mode_ == Mode::lio) {
                                    if (extrinsics_init_.AddPose(DR_motion, lidar_motion, false)) {
                                        if (extrinsics_init_.CalibExRotation()) {
                                            EXTRINSIC_SET_ = true;
                                            SYSTEM_INIT_ = true;
                                            imu_t_lidar_.setZero();
                                            imu_R_lidar_ = extrinsics_init_.GetCalibRot().toRotationMatrix(); 
                                            lidar_trackers_->ResetLocalmap();  
                                            std::cout << SlamLib::color::GREEN << "calibration done ! imu_R_lidar_: " << std::endl
                                                << imu_R_lidar_.matrix() << std::endl << "imu_t_lidar_: " << imu_t_lidar_.transpose() 
                                                << SlamLib::color::RESET << std::endl; 
                                        }  
                                    }
                                }
                            }
                        } else {
                            SYSTEM_INIT_ = true;  
                            Eigen::Quaterniond imu_q_lidar(imu_R_lidar_); 
                            gyro_bias_Estimate_.Initialize(imu_q_lidar, sensor_param_.imu_.gyro_bias_);
                            std::cout << SlamLib::color::GREEN << "估计器初始化成功 ！" << SlamLib::color::RESET << std::endl;
                        }
                    }
                }
                
                ResultInfo<_PointT> result;
                result.time_stamps_ = curr_lidar_data.timestamp_start_;
                result.pose_ = pose_;
                result.predict_pose_ = predict_pose; 
                result.pointcloud_ = curr_lidar_data.feature_data_;
                result.pointcloud_.insert(curr_lidar_data.pointcloud_data_.begin(), curr_lidar_data.pointcloud_data_.end());
                result.pointcloud_["dynamic_points"] = lidar_trackers_->GetDynamicCloud();  
                result.pointcloud_["false_dynamic_points"] = lidar_trackers_->GetFalseDynamicCloud();  
                
                if (lidar_trackers_->HasUpdataLocalMap()) {
                    const auto& local_map = lidar_trackers_->GetLocalMap();  
                    result.pointcloud_map_.insert(local_map.begin(), local_map.end());
                }
                comm::IntraProcess::Server::Instance().Publish("odom_res", result); 
                tt.toc("pub "); 
                lidar_sm_.lock();
                processed_lidar_buf_.pop_front();  
                lidar_sm_.unlock();  
                imu_selected.clear();  
                wheel_odom_selected.clear();   
            }
        }
        // 当没有激光数据时，或者组合导航与激光雷达的外参未知时，需要进行INS
        if (!EXTRINSIC_SET_) {
            if (gnss_buf_.size()) {
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief 根据lidar的运动估计车体的运动线速度  匀速运动模型 
 * @param lidar_motion lidar的运动增量 
 * @param incre_time 时间间隔  
*/
template<typename _PointT>    
void System<_PointT>::estimateBodyLinearVelocity(
        Eigen::Isometry3d const& lidar_motion, double const& incre_time) {
    // 由lidar的运动平移转求解车体的运动平移
    Eigen::Matrix3d body_rot = imu_R_lidar_ * lidar_motion.linear() * imu_R_lidar_.inverse();  
    Eigen::Vector3d body_trans = imu_R_lidar_ * lidar_motion.translation() + 
        imu_t_lidar_ - body_rot * imu_t_lidar_;
    last_velocity_ = curr_velocity_;  
    curr_velocity_ = body_trans / incre_time;  
    filtered_velocity_ = (last_velocity_ + curr_velocity_) / 2;
    // std::cout << "filtered_velocity_ : " << filtered_velocity_.transpose() << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename _PointT>    
bool System<_PointT>::syncSensorData(const double& start_time, const double& end_time,
                                        std::deque<sensor::OdomData>& wheelOdom_container,
                                        std::deque<sensor::ImuData>& imu_container) {
    // std::cout << "syncSensorData" << std::endl;
    bool imu_extract_finish = true; 
    bool wheel_extract_finish = true;
    // 提取轮速数据
    wheel_sm_.lock();
    if (!extractSensorData<sensor::OdomData>(odom_buf_, wheelOdom_container,
            start_time, end_time)) {
        wheel_extract_finish = false; 
    }
    wheel_sm_.unlock();
    // 提取IMU数据
    imu_sm_.lock();
    if (!extractSensorData<sensor::ImuData>(imu_buf_, imu_container,
            start_time, end_time)) {
        imu_extract_finish = false; 
    }
    imu_sm_.unlock();
    return wheel_extract_finish && imu_extract_finish; 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: 从目标容器中提取出时间范围内的数据
 * @details: 处理后，data_cache 的第一个数据就要晚于 start_time 
 * @param data_cache 目标容器
 * @param extracted_container 提取数据放置的容器
 * @param start_time 最早时间
 * @param end_time 最晚时间
 * @return 提取是否完成 
 */    
template<typename _PointT>
template<typename DataT_>
bool System<_PointT>::extractSensorData (
        std::deque<DataT_>& data_cache, std::deque<DataT_>& extracted_container,
        const double& start_time, const double& end_time) {
    // std::cout << std::setprecision(15) << "start_time: " << start_time << std::endl;
    // std::cout << std::setprecision(15) << "end_time" << end_time << std::endl;
    if (start_time == end_time) {
        std::cout << "lidar start_time == end_time " << std::endl;
        return true;  
    }
    if (!extracted_container.empty())  return true;  
    if (!data_cache.empty()) {
        if (data_cache.front().timestamp_ <= start_time) {
            if (data_cache.back().timestamp_ >= end_time) {
                // std::cout << std::setprecision(15) << "data_cache.back().timestamp_" << data_cache.back().timestamp_ << std::endl;
                // 添加第一个头数据 
                DataT_ begin_data;     // 时间戳 <= 激光起始时间戳的第一个数据

                while (data_cache.front().timestamp_  <= start_time) {
                    begin_data = data_cache.front();
                    data_cache.pop_front();  
                }
                // std::cout << std::setprecision(15) << "while push_back begin_data, data_cache size: " << data_cache.size() << 
                //  "data_cache.front().timestamp_: " << data_cache.front().timestamp_ << std::endl;
                // 若这个begin_data 距离 起始时间太远了，那么需要进一步找begin_data
                if (start_time - begin_data.timestamp_ > 1e-3) {
                    // 先直接看下一个数据 是否距离激光起始时间足够近
                    if (data_cache.front().timestamp_ - start_time < 1e-3) {
                        begin_data = data_cache.front();
                        data_cache.pop_front();  
                    } else {
                        // 插值
                        begin_data = Utility::Interpolate(begin_data, data_cache.front(), 
                                                                        begin_data.timestamp_ , data_cache.front().timestamp_, 
                                                                        start_time);
                    }
                }
                // 起始时间戳和激光第一个点的时间戳对齐  
                begin_data.timestamp_ = start_time;  
                extracted_container.push_back(begin_data);   // 放置第一个数据
                // std::cout << "begin_data push_back, data_cache size: " << data_cache.size() << std::endl;
                // 添加中间段的数据
                auto data_ptr = data_cache.begin();  

                for (; data_ptr->timestamp_ <= end_time; ++data_ptr) {
                    extracted_container.push_back(*data_ptr);
                }
                // std::cout << "while push_back mid_data" << std::endl;
                // 如果轮速最后一个数据的时间戳距离laser最后一个点的时间较远  那么向后寻找一个更接近的轮速数据
                if (end_time - extracted_container.back().timestamp_ > 1e-3) {
                    if (data_ptr->timestamp_ - end_time < 1e-3) {
                        extracted_container.push_back(*data_ptr); 
                    } else {
                        // 插值
                        auto end_data = Utility::Interpolate(extracted_container.back(), *data_ptr, 
                            extracted_container.back().timestamp_ , data_ptr->timestamp_, 
                            end_time);
                        extracted_container.push_back(end_data); 
                    }
                }
                // 最后一个轮速数据的时间戳和激光最后一个点的时间戳对齐 
                extracted_container.back().timestamp_ = end_time; 
                // std::cout << "while push_back end_time" << std::endl;
            } else {
                // std::cout << "data_cache.back().timestamp_ < end_time" << std::endl;
                return false;  
            }
        } else {
            // std::cout << "data_cache.front().timestamp_ > start_time" << std::endl;
        }
    } else {
        // std::cout << "data_cache.empty()" << std::endl;
    }
    return true;  
}
}// namespace 


