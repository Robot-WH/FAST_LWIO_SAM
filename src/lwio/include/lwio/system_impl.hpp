
#pragma once 
#include <iomanip> 
#include "SlamLib/tic_toc.hpp"
#include "SlamLib/Common/point_type.h"
#include "lwio/system.h"

namespace lwio {

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename _PointT>
System<_PointT>::System(std::string param_path) : param_path_(param_path) {
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
    // 地面分割 + 聚类
    SlamLib::PCLPtr<_PointT> unground_points(new pcl::PointCloud<_PointT>);
    SlamLib::PCLPtr<_PointT> ground_points(new pcl::PointCloud<_PointT>);
    // segmentation_->Process(curr_data.pointcloud_ptr_, unstable_points, stable_points, ground_points);  
    std::vector<uint32_t> ground_index; 
    ground_detect_->GroundDetect(*filtered_points, ground_index, *ground_points);  // 地面检测 
    tt.toc("点云处理：");

    SlamLib::CloudContainer<_PointT> feature_points;  
    feature_points.ori_points_num = curr_data.pointcloud_ptr_->size();  
    feature_points.timestamp_start_ = curr_data.timestamp_; 
    feature_points.timestamp_end_ = curr_data.end_timestamp_; 
    feature_points.feature_data_.insert(std::make_pair("filtered", filtered_points));  // 用于匹配
    feature_points.pointcloud_data_.insert(std::make_pair("ground_points", ground_points));  // 地面点  
    // 处理好的点云数据 放置到缓存队列中  等待  估计器使用  
    lidar_sm_.lock();
    processed_lidar_buf_.emplace_back(std::move(feature_points));  
    lidar_sm_.unlock();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: 对于IMU数据的处理
 */    
template<typename _PointT>
void System<_PointT>::InputData(sensor::ImuData& data) {
    imu_sm_.lock();
    imu_buf_.push_back(std::move(data));  
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
 * @brief: 对于GNSS数据的处理 
 */            
template<typename _PointT>    
void System<_PointT>::InputData(sensor::WheelOdom& data) {
    odom_sm_.lock();
    odom_buf_.push_back(std::move(data));
    odom_sm_.unlock(); 
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
    std::deque<sensor::WheelOdom> wheel_odom_selected; 

    while (1) {
        std::shared_lock<std::shared_mutex> s_l(lidar_sm_);   // 读锁
        // 如果有待处理的激光数据 
        if (processed_lidar_buf_.size()) {
            auto& curr_lidar_data = processed_lidar_buf_.front();  
            s_l.unlock(); 
            ++wait_time; 
            // 提取出包围lidar的imu数据, 
            if (!EXTRINSIC_GET_ || syncSensorData(curr_lidar_data.timestamp_start_, 
                    curr_lidar_data.timestamp_end_, wheel_odom_selected, imu_selected) || wait_time > 50) {
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
                if (!imu_selected.empty()) {
                    std::cout << SlamLib::color::GREEN << "--------------------- lidar imu fusion !-----------------------" 
                        << SlamLib::color::RESET << std::endl; 
                    std::cout << "imu num: " << imu_selected.size() << std::endl;

                } else {
                    std::cout << SlamLib::color::GREEN << "--------------------- lidar odometry !-----------------------" 
                        << SlamLib::color::RESET << std::endl; 
                }

                if (!wheel_odom_selected.empty()) {
                    std::cout << SlamLib::color::GREEN << "--------------------- lidar imu odom fusion !-----------------------" 
                        << SlamLib::color::RESET << std::endl; 
                    std::cout << "wheel_odom num: " << wheel_odom_selected.size() << std::endl;

                }
                // 去畸变 + 预测  


                /**
                 * @todo 实现EKF融合先验运动学模型
                 */
                // // if (!SYSTEM_INIT_) {
                Eigen::Isometry3d motion = Eigen::Isometry3d::Identity(); 
                // // @TODO 基于EKF的预测
                // TicToc tt;
                lidar_trackers_->Solve(curr_lidar_data, motion);   
                // tt.toc("lidar tracker "); 
                pose_ = pose_ * motion; // 当前帧的绝对运动   
                //     // Eigen::Vector3f p = pose_.translation().cast<float>();
                //     // Eigen::Quaternionf quat(pose_.rotation().cast<float>());
                //     // quat.normalize();
                //     // 进行初始化  
                //     // imu_init_.Initialize(motion, imu_selected);  
                // // }
                // // LidarImuDataPacket<_PointT> measure;  
                // // estimator_->ProcessData(measure);  
                
                ResultInfo<_PointT> result;
                result.time_stamps_ = curr_lidar_data.timestamp_start_;
                result.pose_ = pose_;
                result.pointcloud_ = curr_lidar_data.feature_data_;
                result.pointcloud_.insert(curr_lidar_data.pointcloud_data_.begin(), curr_lidar_data.pointcloud_data_.end());
                result.pointcloud_["dynamic_points"] = lidar_trackers_->GetDynamicCloud();  
                result.pointcloud_["false_dynamic_points"] = lidar_trackers_->GetFalseDynamicCloud();  
                
                if (lidar_trackers_->HasUpdataLocalMap()) {
                    const auto& local_map = lidar_trackers_->GetLocalMap();  
                    result.pointcloud_.insert(local_map.begin(), local_map.end());
                }
                // for (const auto& it : result.pointcloud_) {
                //     std::cout << "name: " << it.first << std::endl;
                // }
                comm::IntraProcess::Server::Instance().Publish("odom_res", result); 

                lidar_sm_.lock();
                processed_lidar_buf_.pop_front();  
                lidar_sm_.unlock();  
                imu_selected.clear();  
                wheel_odom_selected.clear();   
            }

        }
        // 当没有激光数据时，或者组合导航与激光雷达的外参未知时，需要进行INS
        if (!EXTRINSIC_GET_) {
            if (gnss_buf_.size()) {
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename _PointT>    
bool System<_PointT>::syncSensorData(const double& start_time, const double& end_time,
                                        std::deque<sensor::WheelOdom>& wheelOdom_container,
                                        std::deque<sensor::ImuData>& imu_container) {
    // std::cout << "syncSensorData" << std::endl;
    bool imu_extract_finish = true; 
    bool wheel_extract_finish = true;
    // 提取轮速数据
    wheel_sm_.lock();
    if (!extractSensorData<sensor::WheelOdom>(odom_buf_, wheelOdom_container,
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
    if (start_time == end_time) return true;  
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


