/*
 * @Copyright(C): 
 * @FileName: 
 * @Author: lwh
 * @Description:  faster_loam
 *                                  特点： 1、具有 时间范围/空间范围 两种localmap构造方式
 *                                                 2、ivox加速近邻搜索
 *                                                 3、实现特征法和直接法 
 *                                                 4、支持旋转式激光和固态激光
 *                                                 
 */
#include "ros_utils.hpp"
#include "lwio/Sensor/lidar_data.h"
#include "lwio/system.h"
#include "comm/InnerProcessComm.hpp"
#include "SlamLib/tic_toc.hpp"
#include <execution>  // C++ 17 并行算法 
#include <atomic>

constexpr bool PARAM_ESTIMATE = true;  // 外参估计 
using namespace std; 
// using UsedPointT = PointXYZIRDT;
using UsedPointT = PointXYZIRDTC;
// using UsedPointT = pcl::PointXYZI;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class LwioRosWrapper {
public:
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    LwioRosWrapper() {
        // \033[1;32m，\033[0m 终端显示成绿色
        ROS_INFO_STREAM("\033[1;32m----> faster loam node started.\033[0m");
        std::string imu_topic = RosReadParam<string>(node_handle_, "ImuTopic");
        imu_subscriber_ = node_handle_.subscribe(imu_topic, 1000, &LwioRosWrapper::imuHandler,
                                                    this, ros::TransportHints().tcpNoDelay());
        std::string odom_topic = RosReadParam<string>(node_handle_, "OdomTopic");
        wheel_odom_subscriber_ = node_handle_.subscribe(
            odom_topic, 2000, &LwioRosWrapper::OdomCallback, this, ros::TransportHints().tcpNoDelay());
        std::string gnss_topic = RosReadParam<string>(node_handle_, "GnssTopic");
        gnss_subscriber_ = node_handle_.subscribe(gnss_topic, 1000, &LwioRosWrapper::gnssHandler, 
                                                    this, ros::TransportHints().tcpNoDelay());  
        std::string lidar_topic = RosReadParam<string>(node_handle_, "LidarTopic");
        int type = RosReadParam<int>(node_handle_, "LidarType");
        if (type == LidarType::Livox) {
            option_.lidar_type_ = LidarType::Livox;
            lidar_subscriber_ = node_handle_.subscribe(
                lidar_topic, 5, &LwioRosWrapper::LivoxCallBack, this, ros::TransportHints().tcpNoDelay());
            LOG(INFO) << "lidar type: Livox"; 
        } else if (type == LidarType::Velodyne || type == LidarType::Ouster) {
            lidar_subscriber_ = node_handle_.subscribe(
                lidar_topic, 5, &LwioRosWrapper::StandardLidarCallback, this, ros::TransportHints().tcpNoDelay());
            if (type == LidarType::Velodyne) {
                option_.lidar_type_ = LidarType::Velodyne;
                LOG(INFO) << "lidar type: velodyne"; 
            } else if (type == LidarType::Ouster) {
                option_.lidar_type_ = LidarType::Ouster;
                LOG(INFO) << "lidar type: ouster"; 
            }
        } 
        option_.max_range_thresh_ = RosReadParam<float>(node_handle_, "MaxRangeThresh");
        option_.min_range_thresh_ = RosReadParam<float>(node_handle_, "MinRangeThresh");
        option_.param_path_ = RosReadParam<std::string>(node_handle_, "ParamPath");
    
        preprocessed_pointcloud_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
            "preprocessed_pointcloud", 1, this);
        dynamic_pointcloud_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
            "dynamic_pointcloud", 1, this);
        false_dynamic_pointcloud_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
            "/false_dynamic_pointcloud", 1);
        stable_pointcloud_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
            "stable_pointcloud", 1, this);
        ground_pointcloud_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
            "/ground_cloud", 1);
        nonground_pointcloud_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
            "/nonground_cloud", 1);
        outlier_pointcloud_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
            "/outlier_cloud", 1);
        surf_feature_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
            "/surf_feature", 1);
        local_map_filtered_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
            "/filtered_local_map", 1);
        markers_publisher_ = node_handle_.advertise<visualization_msgs::MarkerArray>(
            "/cluster_markers", 10);        // 可视化
        odom_pub_ = node_handle_.advertise<nav_msgs::Odometry>(
            "/fusion_odom", 10);
        predict_odom_pub_ = node_handle_.advertise<nav_msgs::Odometry>(
            "/predict_odom", 10);
        // AccSigmaLimit_ = 0.002;
        system_ptr_.reset(new lwio::System<UsedPointT>(option_.param_path_));
        pose_ = Eigen::Isometry3d::Identity(); 
        comm::IntraProcess::Server::Instance().Subscribe("odom_res",
            &LwioRosWrapper::FusionOdomResultCallback, this);  
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void imuHandler(sensor_msgs::ImuConstPtr const& imu_msg) {
        // 保证队列中 数据的顺序正确 
        // m_buf.lock();
        static double last_imu_t = -1; 
        if (imu_msg->header.stamp.toSec() <= last_imu_t) {
            ROS_WARN("imu message in disorder!");
            return;
        }
        last_imu_t = imu_msg->header.stamp.toSec();
        // 解析IMU数据 
        lwio::sensor::ImuData imu_data;
        // 保存时间戳 
        imu_data.timestamp_ = imu_msg->header.stamp.toSec();
        imu_data.acc_ << imu_msg->linear_acceleration.x, 
                            imu_msg->linear_acceleration.y,
                            imu_msg->linear_acceleration.z;
        imu_data.gyro_ << imu_msg->angular_velocity.x,
                            imu_msg->angular_velocity.y,
                            imu_msg->angular_velocity.z;
        imu_data.rot_.w() = imu_msg->orientation.w;
        imu_data.rot_.x() = imu_msg->orientation.x;
        imu_data.rot_.y() = imu_msg->orientation.y;
        imu_data.rot_.z() = imu_msg->orientation.z;
        static uint32_t t = 0;
        t++;
        int k = t / 1000;
        Eigen::Vector3d w_bias = k * Eigen::Vector3d(0.0001, 0.0001, -0.0001);
        imu_data.gyro_ -= w_bias;
        std::cout << "w_bias: " << w_bias.transpose() << std::endl;
        system_ptr_->InputData(imu_data);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // odom的回调函数
    void OdomCallback(const nav_msgs::Odometry::ConstPtr &odometryMsg) {
        lwio::sensor::OdomData odom;
        odom.timestamp_ = odometryMsg->header.stamp.toSec();
        odom.velocity_ = odometryMsg->twist.twist.linear.x;   // 线速度即为 x轴的速度  
        odom.yaw_angular_vel_ = odometryMsg->twist.twist.angular.z; // yaw的角速度
        // odom位姿     odom为2D，没有z的观测
        // odom.pos_xy_.x() = odometryMsg->pose.pose.position.x;
        // odom.pos_xy_.y() = odometryMsg->pose.pose.position.y;
        // odom.orientation_.w() = odometryMsg->pose.pose.orientation.w;
        // odom.orientation_.x() = odometryMsg->pose.pose.orientation.x;
        // odom.orientation_.y() = odometryMsg->pose.pose.orientation.y;
        // odom.orientation_.z() = odometryMsg->pose.pose.orientation.z;
        system_ptr_->InputData(odom);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void gnssHandler(sensor_msgs::NavSatFixConstPtr const& navsat_msg) {
        // 保证队列中 数据的顺序正确 
        static double last_gnss_t = -1; 
        if (navsat_msg->header.stamp.toSec() <= last_gnss_t) 
        {
            ROS_WARN("gnss message in disorder!");
            return;
        }
        last_gnss_t = navsat_msg->header.stamp.toSec();
        // 解析Gnss数据 
        lwio::sensor::GnssData gnss_data;
        // 保存时间戳 
        gnss_data.timestamp_ = navsat_msg->header.stamp.toSec();
        gnss_data.lla_ << navsat_msg->latitude,
                            navsat_msg->longitude,
                            navsat_msg->altitude;
        gnss_data.cov_ = Eigen::Map<const Eigen::Matrix3d>(navsat_msg->position_covariance.data());
        system_ptr_->InputData(gnss_data);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 旋转式激光的回调  
    void StandardLidarCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
        static double last_timestamp_lidar_ = -1; 
        if (cloud_msg->header.stamp.toSec() < last_timestamp_lidar_) {
            LOG(WARNING) << "激光时间戳错乱!";
            return; 
        }
        SlamLib::time::TicToc tt;
        lwio::LidarData<UsedPointT> extracted_data; 
        last_timestamp_lidar_ = cloud_msg->header.stamp.toSec();
        cloud_header_ = cloud_msg->header;
        // 根据不同的激光雷达进行相应的处理 - 将激光数据从 ros msg中提取出来 
        switch (option_.lidar_type_) {
            case LidarType::Ouster: {
                OusterLidarHandler(cloud_msg, extracted_data);   // 从ros msg 中提取出点云数据
                break;
            }
            case LidarType::Velodyne: {
                VelodyneLidarHandler(cloud_msg, extracted_data);
                break;
            }
            default: {
                LOG(ERROR) << "Error LiDAR Type";
                break;
            }
        }
        
        system_ptr_->InputData(extracted_data); 
        tt.toc("StandardLidarCallback ");
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // livox激光雷达的回调  
    void LivoxCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
        static double last_timestamp_lidar_ = -1; 
        if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
            LOG(WARNING) << "激光时间戳错乱!";
        }
        last_timestamp_lidar_ = msg->header.stamp.toSec();
        lwio::LidarData<UsedPointT> extracted_data; 
        LivoxHandler(msg, extracted_data);
        // 点云处理 - 包括预处理、去畸变、聚类、特征提取等
        system_ptr_->InputData(extracted_data); 
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void LivoxHandler(const livox_ros_driver::CustomMsg::ConstPtr& msg, 
            lwio::LidarData<UsedPointT>& extracted_data) {
        extracted_data.timestamp_ = msg->header.stamp.toSec();
        int plsize = msg->point_num;
        extracted_data.pointcloud_ptr_->reserve(plsize);
        pcl::PointCloud<UsedPointT> cloud_full;
        cloud_full.resize(plsize);
        std::vector<bool> is_valid_pt(plsize, false);
        std::vector<uint> index(plsize - 1);
        for (uint i = 0; i < plsize - 1; ++i) {
            index[i] = i + 1;  // 从1开始
        }
        float max_range_thresh_2 = option_.max_range_thresh_ * option_.max_range_thresh_;
        float min_range_thresh_2 = option_.min_range_thresh_ * option_.min_range_thresh_;
        UsedPointT point; 
        // 并行加速  实测  加速：1.57ms   1.54ms   1.41ms    1.50ms    1.41ms    avg:1.48
        //                          不加速 ：1.78ms   1.67ms    1.57ms    1.94ms    1.7ms   avg: 1.73
        std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const uint &i) {
        //std::for_each(index.begin(), index.end(), [&](const uint &i) {
            if (((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00)) {
                // 降采样
                if (i % option_.simple_point_filter_res_ == 0) {
                    // 点与点之间要有一定的距离 
                    if ((abs(msg->points[i].x - msg->points[i - 1].x) > 1e-7) ||
                        (abs(msg->points[i].y - msg->points[i - 1].y) > 1e-7) ||
                        (abs(msg->points[i].z - msg->points[i - 1].z) > 1e-7)) {
                            // 距离滤波
                            double range = msg->points[i].x * msg->points[i].x + 
                                                            msg->points[i].y * msg->points[i].y + 
                                                            msg->points[i].z * msg->points[i].z;
                            if (range > min_range_thresh_2 && range < max_range_thresh_2) {
                                cloud_full[i].x = msg->points[i].x;
                                cloud_full[i].y = msg->points[i].y;
                                cloud_full[i].z = msg->points[i].z;
                                cloud_full[i].intensity = msg->points[i].reflectivity;
                                // use curvature as time of each laser points, curvature unit: ms
                                cloud_full[i].time = msg->points[i].offset_time / float(1000000);  
                                is_valid_pt[i] = true;
                            }
                    }
                }
            }
        });
        for (uint i = 1; i < plsize; i++) {
            if (is_valid_pt[i]) {
                extracted_data.pointcloud_ptr_->points.push_back(cloud_full[i]);
            }
        }
        // LOG(INFO) <<"extracted_data num:"<<extracted_data.pointcloud_ptr_->size(); 
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief: 将ouster雷达的数据从 ros msg中提取出来，并进行降采样、距离滤波  
     */        
    void OusterLidarHandler(const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
            lwio::LidarData<UsedPointT>& extracted_data) {
        extracted_data.timestamp_ = cloud_msg->header.stamp.toSec();
        pcl::PointCloud<ousterPoint> pl_orig;
        pcl::fromROSMsg<ousterPoint>(*cloud_msg, pl_orig);
        // 去除Nan
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud<ousterPoint>(pl_orig, pl_orig, indices);
        float max_range_thresh_2 = option_.max_range_thresh_ * option_.max_range_thresh_;
        float min_range_thresh_2 = option_.min_range_thresh_ * option_.min_range_thresh_;

        int plsize = pl_orig.points.size();
        extracted_data.pointcloud_ptr_->reserve(plsize);

        for (int i = 0; i < plsize; i++) {
            if (i % option_.simple_point_filter_res_ != 0) continue;   

            double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y +
                        pl_orig.points[i].z * pl_orig.points[i].z;

            if (range < min_range_thresh_2 || range > max_range_thresh_2) {
                continue;
            }

            UsedPointT added_pt;
            added_pt.x = pl_orig.points[i].x;
            added_pt.y = pl_orig.points[i].y;
            added_pt.z = pl_orig.points[i].z;
            added_pt.ring = pl_orig.points[i].ring;
            added_pt.intensity = pl_orig.points[i].intensity;
            added_pt.time = pl_orig.points[i].t / 1e6;  // curvature unit: ms
            added_pt.range = sqrt(range); 
            extracted_data.pointcloud_ptr_->points.push_back(added_pt);
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief: 将velodyne雷达的数据从 ros msg中提取出来，并进行降采样、距离滤波  
     */        
    void VelodyneLidarHandler(const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
                lwio::LidarData<UsedPointT>& extracted_data) {
        pcl::PointCloud<velodynePoint> pl_orig;
        // std::cout << "fromROSMsg..." << std::endl;
        pcl::fromROSMsg<velodynePoint>(*cloud_msg, pl_orig);
        // std::cout << "fromROSMsg done" << std::endl;
        // 去除Nan
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud<velodynePoint>(pl_orig, pl_orig, indices);
        float max_range_thresh_2 = option_.max_range_thresh_ * option_.max_range_thresh_;
        float min_range_thresh_2 = option_.min_range_thresh_ * option_.min_range_thresh_;
        int plsize = pl_orig.points.size();
        extracted_data.timestamp_ = cloud_msg->header.stamp.toSec();
        // extracted_data.end_timestamp_ = extracted_data.timestamp_ + 
        //     (pl_orig.points[plsize - 1].time - pl_orig.points[0].time);
        extracted_data.end_timestamp_ = extracted_data.timestamp_ + 0.0518;

        // extracted_data.pointcloud_ptr_->resize(plsize);
        extracted_data.pointcloud_ptr_->reserve(plsize);

        // #pragma omp parallel for    // 并行加速为啥没用
        for (int i = 0; i < plsize; ++i) {
            // 实测发现  下面这个距离计算与距离滤波对耗时没有什么影响 
            if (i % option_.simple_point_filter_res_ != 0) continue;   

            double range = pl_orig.points[i].x * pl_orig.points[i].x + 
                                            pl_orig.points[i].y * pl_orig.points[i].y +
                                            pl_orig.points[i].z * pl_orig.points[i].z;

            if (range < min_range_thresh_2 || range > max_range_thresh_2) {
                continue;
            }

            UsedPointT added_pt;
            added_pt.x = pl_orig.points[i].x;
            added_pt.y = pl_orig.points[i].y;
            added_pt.z = pl_orig.points[i].z;
            added_pt.ring = pl_orig.points[i].ring;
            added_pt.intensity = pl_orig.points[i].intensity;
            added_pt.time = pl_orig.points[i].time - pl_orig.points[0].time;  // curvature unit: ms
            added_pt.range = sqrt(range); 
            
            extracted_data.pointcloud_ptr_->points.push_back(std::move(added_pt));

            // extracted_data.pointcloud_ptr_->points[i].x = pl_orig.points[i].x;
            // extracted_data.pointcloud_ptr_->points[i].y = pl_orig.points[i].y;
            // extracted_data.pointcloud_ptr_->points[i].z = pl_orig.points[i].z;
            // extracted_data.pointcloud_ptr_->points[i].ring = pl_orig.points[i].ring;
            // extracted_data.pointcloud_ptr_->points[i].intensity = pl_orig.points[i].intensity;
            // extracted_data.pointcloud_ptr_->points[i].time = pl_orig.points[i].time - pl_orig.points[0].time;  // curvature unit: ms
            // temp_points->points[i].range = sqrt(range); 
        }
        // std::cout << "VelodyneLidarHandler done" << std::endl;
    }

    /**
     * @brief: 接收融合算法计算的结果     
     */
    void FusionOdomResultCallback(const lwio::ResultInfo<UsedPointT>& data) {
        // 发布融合位姿
        Eigen::Vector3f p = data.pose_.translation().cast<float>();
        Eigen::Quaternionf quat(data.pose_.rotation().cast<float>());
        quat.normalize();
        ros::Time stamp = ros::Time(data.time_stamps_);  
        PublishTF(p, quat, stamp, "odom", "base"); 

        nav_msgs::Odometry odom;                            
        odom.header.stamp = stamp;
        odom.header.frame_id = "odom";       // odom坐标    /odom
        odom.child_frame_id = "base";        // /lidar_odom
        odom.pose.pose.position.x = p[0];
        odom.pose.pose.position.y = p[1];
        odom.pose.pose.position.z = p[2];
        // 构造四元数   ROS信息
        geometry_msgs::Quaternion odom_quat;
        odom_quat.w = quat.w();
        odom_quat.x = quat.x();
        odom_quat.y = quat.y();
        odom_quat.z = quat.z();
        odom.pose.pose.orientation = odom_quat;  
        odom_pub_.publish(odom);

        //  发布预测位姿   用于调试
        p = data.predict_pose_.translation().cast<float>();
        quat = Eigen::Quaternionf(data.predict_pose_.rotation().cast<float>());
        quat.normalize();
        PublishTF(p, quat, stamp, "odom", "predict_base");

        odom.child_frame_id = "predict_base";        // /lidar_odom
        odom.pose.pose.position.x = p[0];
        odom.pose.pose.position.y = p[1];
        odom.pose.pose.position.z = p[2];
        // 构造四元数   ROS信息
        odom_quat.w = quat.w();
        odom_quat.x = quat.x();
        odom_quat.y = quat.y();
        odom_quat.z = quat.z();
        odom.pose.pose.orientation = odom_quat;  
        predict_odom_pub_.publish(odom);

        // 发布点云
        sensor_msgs::PointCloud2 laserCloudTemp;
        if (preprocessed_pointcloud_publisher_.getNumSubscribers() != 0) {
            pcl::toROSMsg(*data.pointcloud_.at("filtered"), laserCloudTemp);
            laserCloudTemp.header.stamp = stamp;
            laserCloudTemp.header.frame_id = "base";
            preprocessed_pointcloud_publisher_.publish(laserCloudTemp);
        }

        if (ground_pointcloud_publisher_.getNumSubscribers() != 0) {
            pcl::toROSMsg(*data.pointcloud_.at("ground_points"), laserCloudTemp);
            laserCloudTemp.header.stamp = stamp;
            laserCloudTemp.header.frame_id = "base";
            ground_pointcloud_publisher_.publish(laserCloudTemp);
        }

        if (stable_pointcloud_publisher_.getNumSubscribers() != 0) {
            pcl::toROSMsg(*data.pointcloud_.at("stable_points"), laserCloudTemp);
            laserCloudTemp.header.stamp = stamp;
            laserCloudTemp.header.frame_id = "base";
            stable_pointcloud_publisher_.publish(laserCloudTemp);
        }

        if (dynamic_pointcloud_publisher_.getNumSubscribers() != 0) {
            pcl::toROSMsg(*data.pointcloud_.at("dynamic_points"), laserCloudTemp);
            laserCloudTemp.header.stamp = stamp;
            laserCloudTemp.header.frame_id = "base";
            dynamic_pointcloud_publisher_.publish(laserCloudTemp);
        }

        if (false_dynamic_pointcloud_publisher_.getNumSubscribers() != 0) {
            pcl::toROSMsg(*data.pointcloud_.at("false_dynamic_points"), laserCloudTemp);
            laserCloudTemp.header.stamp = stamp;
            laserCloudTemp.header.frame_id = "base";
            false_dynamic_pointcloud_publisher_.publish(laserCloudTemp);
        }

        if (data.pointcloud_map_.find("filtered") != data.pointcloud_map_.end()) {
            publishCloud<UsedPointT>( &local_map_filtered_publisher_,    // 发布该点云的话题 
                                data.pointcloud_map_.at("filtered"),   // 点云数据   
                                stamp, "odom");     
        }
    }


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void publishLidarScan() {
        // sensor_msgs::PointCloud2 laserCloudTemp;
        // if (preprocessed_pointcloud_publisher_.getNumSubscribers() != 0) {
        //     pcl::toROSMsg(*preprocessed_cloud_, laserCloudTemp);
        //     laserCloudTemp.header.stamp = cloud_header_.stamp;
        //     laserCloudTemp.header.frame_id = "lidar";
        //     preprocessed_pointcloud_publisher_.publish(laserCloudTemp);
        // }

        // // if (nonground_pointcloud_publisher_.getNumSubscribers() != 0) {
        // //     pcl::PointCloud<UsedPointT> const& nonground_points = 
        // //         segmentation_->GetNonGroundPoints(); 
        // //     pcl::toROSMsg(nonground_points, laserCloudTemp);
        // //     laserCloudTemp.header.stamp = cloud_header_.stamp;
        // //     laserCloudTemp.header.frame_id = "base_link";
        // //     nonground_pointcloud_publisher_.publish(laserCloudTemp);
        // // }


        // // if (outlier_pointcloud_publisher_.getNumSubscribers() != 0) {
        // //     pcl::PointCloud<UsedPointT> const& outlier_points = 
        // //         segmentation_->GetOutlierPoints(); 
        // //     pcl::toROSMsg(outlier_points, laserCloudTemp);
        // //     laserCloudTemp.header.stamp = cloud_header_.stamp;
        // //     laserCloudTemp.header.frame_id = "base_link";
        // //     outlier_pointcloud_publisher_.publish(laserCloudTemp);
        // // }

        // // if (edge_feature_publisher_.getNumSubscribers() != 0) {
        // //     CloudContainer<UsedPointT> const& processed_points = processed_points_.back();
        // //     if (processed_points.pointcloud_data_.find("loam_edge") != processed_points.pointcloud_data_.end()) {
        // //         pcl::toROSMsg(*processed_points.pointcloud_data_.at("loam_edge"), laserCloudTemp);
        // //         laserCloudTemp.header.stamp = cloud_header_.stamp;
        // //         laserCloudTemp.header.frame_id = "base_link";
        // //         edge_feature_publisher_.publish(laserCloudTemp);
        // //     }
        // // }

        // // if (surf_feature_publisher_.getNumSubscribers() != 0) {
        // //     CloudContainer<UsedPointT> const& processed_points = processed_points_.back();
        // //     if (processed_points.pointcloud_data_.find("loam_surf") != processed_points.pointcloud_data_.end()) {
        // //         pcl::toROSMsg(*processed_points.pointcloud_data_.at("loam_surf"), laserCloudTemp);
        // //         laserCloudTemp.header.stamp = cloud_header_.stamp;
        // //         laserCloudTemp.header.frame_id = "base_link";
        // //         surf_feature_publisher_.publish(laserCloudTemp);
        // //     }
        // // }
        // if (surf_feature_publisher_.getNumSubscribers() != 0) {
        //     SlamLib::CloudContainer<UsedPointT> const& processed_points = processed_points_.back();
        //     if (processed_points.pointcloud_data_.find("bad_point") != processed_points.pointcloud_data_.end()) {
        //         pcl::toROSMsg(*processed_points.pointcloud_data_.at("bad_point"), laserCloudTemp);
        //         laserCloudTemp.header.stamp = cloud_header_.stamp;
        //         laserCloudTemp.header.frame_id = "base_link";
        //         surf_feature_publisher_.publish(laserCloudTemp);
        //     }
        // }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void publishLocalMapCloud(ros::Time const& stamp) {
        // lwio::LidarTracker<UsedPointT>::LocalMapContainer local_maps = lidar_trackers_->GetLocalMap();
        // for (auto const& iter : local_maps) {
        //     if (iter.first == "loam_edge") {
        //         // publishCloud( &pubLocalMapEdge[id],    // 发布该点云的话题 
        //         //                                 iter->second,   // 边缘特征   
        //         //                                 stamp, odom_frame);     
        //     } else if (iter.first == "loam_surf") {
        //         // publishCloud( &pubLocalMapSurf[id],    // 发布该点云的话题 
        //         //                                 iter->second,   // 点云数据   
        //         //                                 stamp, odom_frame);     
        //     } else if (iter.first == "filtered") {     // 滤波后的 
        //         publishCloud<UsedPointT>( &local_map_filtered_publisher_,    // 发布该点云的话题 
        //                                         iter.second,   // 点云数据   
        //                                         stamp, "odom");     
        //     }
        // }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void pubMarker() {
        // // if (markers_publisher_.getNumSubscribers()) {
        //     std::vector<lwio::PointCloudSegmentation<UsedPointT>::ClusterInfo> const& cluster_info 
        //         = segmentation_->GetClusterInfo();
        //     // std::vector<PointCloudRangeImageSegmentation<UsedPointT>::ClusterInfo> const& cluster_info 
        //     //     = segmentation_->GetClusterInfo();
            
        //     int size = cluster_info.size(); 

        //     visualization_msgs::MarkerArray markers;
        //     markers.markers.resize(size);
        //     for (int i = 0; i < size; i++) {
        //         // sphere
        //         visualization_msgs::Marker& sphere_marker = markers.markers[i];
        //         sphere_marker.header.frame_id = "base_link";
        //         sphere_marker.header.stamp = cloud_header_.stamp;
        //         sphere_marker.ns = "cluster";
        //         sphere_marker.id = i;
        //         sphere_marker.type = visualization_msgs::Marker::SPHERE;

        //         sphere_marker.pose.position.x = cluster_info[i].centre_.x();
        //         sphere_marker.pose.position.y = cluster_info[i].centre_.y();
        //         sphere_marker.pose.position.z = cluster_info[i].centre_.z();

        //         sphere_marker.pose.orientation.w = 1.0;
        //         sphere_marker.scale.x = sphere_marker.scale.y = 
        //             sphere_marker.scale.z = 2 * cluster_info[i].direction_max_value_[2];
        //         //sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = 1;
        //         sphere_marker.color.r = 1.0;
        //         sphere_marker.color.a = 0.3;
        //     }
        //     markers_publisher_.publish(markers);
        // // }
    }


private:
    enum LidarType {Livox = 0, Velodyne, Ouster};
    struct Option {
        LidarType lidar_type_; 
        std::string param_path_; 
        uint16_t simple_point_filter_res_ = 1;
        float min_range_thresh_ = -1;
        float max_range_thresh_ = -1;  
    };
    struct LaserInfo {
        std::vector<double> index_cos_; // 保存下来雷达各个角度的cos值
        std::vector<double> index_sin_; // 保存下来雷达各个角度的sin值
        float angle_increment_;  // 激光点角度增量  
        double time_increment_;  
    }laser_info_;

    Option option_; 
    ros::NodeHandle node_handle_;  
    ros::Subscriber lidar_subscriber_;
    ros::Subscriber imu_subscriber_;     
    ros::Subscriber wheel_odom_subscriber_;
    ros::Subscriber gnss_subscriber_;     
    ros::Publisher preprocessed_pointcloud_publisher_;
    ros::Publisher stable_pointcloud_publisher_;  
    ros::Publisher dynamic_pointcloud_publisher_;  
    ros::Publisher ground_pointcloud_publisher_;
    ros::Publisher nonground_pointcloud_publisher_;
    ros::Publisher outlier_pointcloud_publisher_;
    ros::Publisher false_dynamic_pointcloud_publisher_;
    ros::Publisher surf_feature_publisher_;
    ros::Publisher local_map_filtered_publisher_;
    ros::Publisher markers_publisher_; // 可视化
    ros::Publisher odom_pub_; 
    ros::Publisher predict_odom_pub_;

    std_msgs::Header cloud_header_;

    std::string laser_frame = ""; 
    std::mutex laser_mt_;  

    std::unique_ptr<lwio::System<UsedPointT>> system_ptr_; 
    Eigen::Isometry3d pose_;    // 运动
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "faster_loam");
    LwioRosWrapper node;
    ros::spin(); 
    return (0);
}







