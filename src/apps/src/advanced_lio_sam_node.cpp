/*
 * @Copyright(C): 
 * @Author: lwh
 * @Description:  fast_ligo (快速的激光-imu-gnss融合里程计)    
 *                                  特点： 1、基于滤波器(ESKF/IESKF)融合的LIO
 *                                                 2、使用isam融合GNSS和LIO
 *                                                 3、基于faster_loam开发
 */
#define PCL_NO_PRECOMPILE

#include "ros_utils.hpp"
#include "Sensor/sensor.hpp"
#include "Sensor/lidar_data_type.h"
#include "Common/pcl_type.h"
#include "Common/color.hpp"
#include "Algorithm/PointClouds/Process/Preprocess/RotaryLidarPreProcess.hpp"
#include "Algorithm/PointClouds/Process/Segmentation/PointCloudRangeImageSegmentation.hpp"
#include "Algorithm/PointClouds/Process/Segmentation/PointCloudSegmentation.hpp"
#include "Algorithm/PointClouds/Process/FeatureExtract/LOAMFeatureProcessor_base.hpp"
#include "LidarTracker/LidarTracker.hpp"
#include <execution>  // C++ 17 并行算法 
#include <atomic>

constexpr bool PARAM_ESTIMATE = true;  // 外参估计 
using namespace std; 
using namespace Slam3D; 
using namespace Slam3D::Algorithm; 
using UsedPointT = PointXYZIRTD;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class LIGORosNode {
public:
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    LIGORosNode() {
        // \033[1;32m，\033[0m 终端显示成绿色
        ROS_INFO_STREAM("\033[1;32m----> faster loam node started.\033[0m");
        std::string lidar_topic = RosReadParam<string>(node_handle_, "LidarTopic");
        std::string imu_topic = RosReadParam<string>(node_handle_, "ImuTopic");
        std::string gps_topic = RosReadParam<string>(node_handle_, "GnssTopic");
        imu_subscriber_ = node_handle_.subscribe(imu_topic, 1000, &LIGORosNode::imuHandler,
                                                    this, ros::TransportHints().tcpNoDelay());
        gnss_subscriber_ = node_handle_.subscribe(gps_topic, 1000, &LIGORosNode::gnssHandler, 
                                                    this, ros::TransportHints().tcpNoDelay());  
        int type = RosReadParam<int>(node_handle_, "LidarType");
        if (type == LidarType::Livox) {
            option_.lidar_type_ = LidarType::Livox;   
            lidar_subscriber_ = node_handle_.subscribe(
                lidar_topic, 5, &LIGORosNode::LivoxCallBack, this, ros::TransportHints().tcpNoDelay());
            LOG(INFO) << "lidar type: Livox"; 
        } else if (type == LidarType::Velodyne || type == LidarType::Ouster) {
            lidar_subscriber_ = node_handle_.subscribe(
                lidar_topic, 5, &LIGORosNode::StandardLidarCallback, this, ros::TransportHints().tcpNoDelay());
            if (type == LidarType::Velodyne) {
                option_.lidar_type_ = LidarType::Velodyne;
                LOG(INFO) << "lidar type: velodyne"; 
            } else if (type == LidarType::Ouster) {
                option_.lidar_type_ = LidarType::Ouster;
                LOG(INFO) << "lidar type: ouster"; 
            }
        } 
        option_.num_scans_ = RosReadParam<int>(node_handle_, "LidarLineNum");
        option_.max_range_thresh_ = RosReadParam<float>(node_handle_, "MaxRangeThresh");
        option_.min_range_thresh_ = RosReadParam<float>(node_handle_, "MinRangeThresh");
        option_.param_path_ = RosReadParam<std::string>(node_handle_, "ParamPath");
    
        preprocessed_pointcloud_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
            "preprocessed_pointcloud", 1, this);
        unstable_pointcloud_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
            "unstable_pointcloud", 1, this);
        stable_pointcloud_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
            "stable_pointcloud", 1, this);
        ground_pointcloud_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
            "/ground_cloud", 1);
        nonground_pointcloud_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
            "/nonground_cloud", 1);
        outlier_pointcloud_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
            "/outlier_cloud", 1);
        edge_feature_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
            "/edge_feature", 1);
        surf_feature_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
            "/surf_feature", 1);
        local_map_filtered_publisher_ = node_handle_.advertise<sensor_msgs::PointCloud2>(
            "/filtered_local_map", 1);
        markers_publisher_ = node_handle_.advertise<visualization_msgs::MarkerArray>(
            "/cluster_markers", 10);        // 可视化
        // AccSigmaLimit_ = 0.002;
        Init();  
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void Init() {
        if (option_.lidar_type_ == LidarType::Livox) {
            preprocess_.reset(new LidarPreProcess<UsedPointT>(option_.param_path_));    
        } else if (option_.lidar_type_ == LidarType::Velodyne || option_.lidar_type_ == LidarType::Ouster) {
            preprocess_.reset(new RotaryLidarPreProcess<UsedPointT>(option_.param_path_)); 
        } 
        // 构造tracker  
        lidar_trackers_.reset(new LidarTracker<UsedPointT>(option_.param_path_)); 
        pose_ = Eigen::Isometry3d::Identity(); 
        
        // LOAMFeatureExtractor<UsedPointT, UsedPointT>::Option featureExtract_option;
        // featureExtract_option.lidar_freq_ = 19.3;
        // //featureExtract_option.lidar_freq_ = 10;
        // featureExtract_option.horizon_angle_resolution_ = 0.386;
        // //featureExtract_option.horizon_angle_resolution_ = 0.2;
        // featureExtract_option.edge_thresh_ = 1; 
        // feature_extractor_.reset(new LOAMFeatureExtractor<UsedPointT, UsedPointT>(featureExtract_option));
        // PointCloudRangeImageSegmentation<UsedPointT>::Option seg_option; 
        // seg_option.horizon_angle_resolution_ = 0.386;
        // seg_option.segment_angle_thresh_ = 0.05;
        // segmentation_.reset(new PointCloudRangeImageSegmentation<UsedPointT>(seg_option)); 
        // PointCloudSegmentation<UsedPointT>::Option seg_option; 
        // seg_option.horizon_angle_resolution_ = 0.386;
        // seg_option.segment_angle_thresh_ = 0.05;
        // segmentation_.reset(new PointCloudSegmentation<UsedPointT>(seg_option)); 
        // PointCloudSegmentation<UsedPointT>::Option seg_option; 
        // segmentation_.reset(new PointCloudSegmentation<UsedPointT>(seg_option)); 
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void imuHandler( sensor_msgs::ImuConstPtr const& imu_msg) {
        // 保证队列中 数据的顺序正确 
        // m_buf.lock();
        static double last_imu_t = -1; 
        if (imu_msg->header.stamp.toSec() <= last_imu_t) {
            ROS_WARN("imu message in disorder!");
            return;
        }
        last_imu_t = imu_msg->header.stamp.toSec();
        // 解析IMU数据 
        ImuDataPtr imu_data_ptr = std::make_shared<ImuData>();
        // 保存时间戳 
        imu_data_ptr->timestamp_ = imu_msg->header.stamp.toSec();
        imu_data_ptr->acc_ << imu_msg->linear_acceleration.x, 
                            imu_msg->linear_acceleration.y,
                            imu_msg->linear_acceleration.z;
        imu_data_ptr->gyro_ << imu_msg->angular_velocity.x,
                            imu_msg->angular_velocity.y,
                            imu_msg->angular_velocity.z;
        imu_data_ptr->rot_.w() = imu_msg->orientation.w;
        imu_data_ptr->rot_.x() = imu_msg->orientation.x;
        imu_data_ptr->rot_.y() = imu_msg->orientation.y;
        imu_data_ptr->rot_.z() = imu_msg->orientation.z;
        imu_buf_.push(imu_data_ptr);
        // m_buf.unlock();
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void gnssHandler(sensor_msgs::NavSatFixConstPtr const& navsat_msg) {
        // 保证队列中 数据的顺序正确 
        // m_buf.lock();
        // static double last_gnss_t = -1; 
        // if (navsat_msg->header.stamp.toSec() <= last_gnss_t) {
        //     ROS_WARN("gnss message in disorder!");
        //     return;
        // }
        // last_gnss_t = navsat_msg->header.stamp.toSec();
        // // 解析Gnss数据 
        // Sensor::GnssDataPtr gnss_data_ptr = std::make_shared<Sensor::GnssData>();
        // // 保存时间戳 
        // gnss_data_ptr->timestamp = navsat_msg->header.stamp.toSec();
        // gnss_data_ptr->lla << navsat_msg->latitude,
        //                     navsat_msg->longitude,
        //                     navsat_msg->altitude;
        // gnss_data_ptr->cov = Eigen::Map<const Eigen::Matrix3d>(navsat_msg->position_covariance.data());
        // gnss_buf.push(gnss_data_ptr);
        // m_buf.unlock();
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 旋转式激光的回调  
    void StandardLidarCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
        static double last_timestamp_lidar_ = -1; 
        if (cloud_msg->header.stamp.toSec() < last_timestamp_lidar_) {
            LOG(WARNING) << "激光时间戳错乱!";
            return; 
        }
        LidarData<UsedPointT> extracted_data; 
        last_timestamp_lidar_ = cloud_msg->header.stamp.toSec();
        cloud_header_ = cloud_msg->header;
        // 根据不同的激光雷达进行相应的处理 - 将激光数据从 ros msg中提取出来 
        switch (option_.lidar_type_) {
            case LidarType::Ouster:
            {
                OusterLidarHandler(cloud_msg, extracted_data);   // 从ros msg 中提取出点云数据
                PointCloudProcess(extracted_data); 
                break;
            }
            case LidarType::Velodyne:
            {
                VelodyneLidarHandler(cloud_msg, extracted_data);
                PointCloudProcess(extracted_data); 
                break;
            }
            default:
            {
                LOG(ERROR) << "Error LiDAR Type";
                break;
            }
        }
        
        // // 点云发布
        // publishLidarScan();
        // //pubMarker(); 
        // processed_points_.pop_front(); 
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // livox激光雷达的回调  
    void LivoxCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
        static double last_timestamp_lidar_ = -1; 
        if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
            LOG(WARNING) << "激光时间戳错乱!";
        }
        last_timestamp_lidar_ = msg->header.stamp.toSec();
        LidarData<UsedPointT> extracted_data; 
        TicToc tt; 
        LivoxHandler(msg, extracted_data);
        tt.toc("LivoxHandler ");
        tt.tic(); 
        // static int i = 500;
        // static double sum = 0; 
        // if (i > 0) {
        //     i--;
        //     sum += time; 
        // } else {
        //     LOG(INFO) << "avg time:"<<sum / 500; 
        // }
        // 点云处理 - 包括预处理、去畸变、聚类、特征提取等
        PointCloudProcess(extracted_data); 
        tt.toc("Process "); 
        // 点云发布
        //publishLidarScan();
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void LivoxHandler(const livox_ros_driver::CustomMsg::ConstPtr &msg, 
            LidarData<UsedPointT> &extracted_data) {
        extracted_data.timestamp = msg->header.stamp.toSec();
        int plsize = msg->point_num;
        extracted_data.point_cloud->reserve(plsize);
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
            if ((msg->points[i].line < option_.num_scans_) &&
                    ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00)) {
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
                extracted_data.point_cloud->points.push_back(cloud_full[i]);
            }
        }
        // LOG(INFO) <<"extracted_data num:"<<extracted_data.point_cloud->size(); 
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief: 将ouster雷达的数据从 ros msg中提取出来，并进行降采样、距离滤波  
     */        
    void OusterLidarHandler(const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
            LidarData<UsedPointT>& extracted_data) {
        extracted_data.timestamp = cloud_msg->header.stamp.toSec();
        pcl::PointCloud<ousterPoint> pl_orig;
        pcl::fromROSMsg<ousterPoint>(*cloud_msg, pl_orig);
        // 去除Nan
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud<ousterPoint>(pl_orig, pl_orig, indices);
        float max_range_thresh_2 = option_.max_range_thresh_ * option_.max_range_thresh_;
        float min_range_thresh_2 = option_.min_range_thresh_ * option_.min_range_thresh_;

        for (int i = 0; i < pl_orig.points.size(); i++) {
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
            extracted_data.point_cloud->points.push_back(added_pt);
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief: 将velodyne雷达的数据从 ros msg中提取出来，并进行降采样、距离滤波  
     */        
    void VelodyneLidarHandler(const sensor_msgs::PointCloud2ConstPtr &cloud_msg,
            LidarData<UsedPointT>& extracted_data) {
        extracted_data.timestamp = cloud_msg->header.stamp.toSec();
        pcl::PointCloud<velodynePoint> pl_orig;
        pcl::fromROSMsg<velodynePoint>(*cloud_msg, pl_orig);
        // 去除Nan
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud<velodynePoint>(pl_orig, pl_orig, indices);
        float max_range_thresh_2 = option_.max_range_thresh_ * option_.max_range_thresh_;
        float min_range_thresh_2 = option_.min_range_thresh_ * option_.min_range_thresh_;
        for (int i = 0; i < pl_orig.points.size(); i++) {
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
            added_pt.time = pl_orig.points[i].time * 1e-3;  // curvature unit: ms
            added_pt.range = sqrt(range); 
            extracted_data.point_cloud->points.push_back(added_pt);
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief: 点云处理  1、预处理  2、去畸变   3、特征提取 
     * @param ori_data 从ros msg中提取出来的点云数据
     */        
    void PointCloudProcess(LidarData<UsedPointT>& data) {
        TicToc tt; 
        // 预处理
        //LOG(INFO) << "before process, point num:"<<data.point_cloud->size();
        preprocess_->Process(data); 
        preprocessed_cloud_ = data.point_cloud; 
        // LOG(INFO) << "after process, point num:"<<data.point_cloud->size();
        // 去畸变
        // 特征提取 / 直接法 
        CloudContainer<UsedPointT> feature_points;  
        feature_points. time_stamp_ = data.timestamp; 
        feature_points.pointcloud_data_.insert(make_pair("filtered", std::move(data.point_cloud)));  
        // try {
        //     feature_extractor_->Extract(data, feature_points); 
        // } catch(const char* e) {
        //     std::cerr <<common::RED<< e << common::RESET<<'\n';
        //     return; 
        // }
        // 处理好的点云数据 放置到缓存队列中  等待  估计器使用  
        processed_points_.push_back(std::move(feature_points));  
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void publishLidarScan() {
        sensor_msgs::PointCloud2 laserCloudTemp;
        if (preprocessed_pointcloud_publisher_.getNumSubscribers() != 0) {
            pcl::toROSMsg(*preprocessed_cloud_, laserCloudTemp);
            laserCloudTemp.header.stamp = cloud_header_.stamp;
            laserCloudTemp.header.frame_id = "lidar";
            preprocessed_pointcloud_publisher_.publish(laserCloudTemp);
        }

        // if (nonground_pointcloud_publisher_.getNumSubscribers() != 0) {
        //     pcl::PointCloud<UsedPointT> const& nonground_points = 
        //         segmentation_->GetNonGroundPoints(); 
        //     pcl::toROSMsg(nonground_points, laserCloudTemp);
        //     laserCloudTemp.header.stamp = cloud_header_.stamp;
        //     laserCloudTemp.header.frame_id = "base_link";
        //     nonground_pointcloud_publisher_.publish(laserCloudTemp);
        // }

        // if (ground_pointcloud_publisher_.getNumSubscribers() != 0) {
        //     pcl::PointCloud<UsedPointT> const& ground_points = 
        //         segmentation_->GetGroundPoints(); 
        //     pcl::toROSMsg(ground_points, laserCloudTemp);
        //     laserCloudTemp.header.stamp = cloud_header_.stamp;
        //     laserCloudTemp.header.frame_id = "base_link";
        //     ground_pointcloud_publisher_.publish(laserCloudTemp);
        // }

        // if (stable_pointcloud_publisher_.getNumSubscribers() != 0) {
        //     pcl::PointCloud<UsedPointT> const& stable_points = 
        //         segmentation_->GetStablePoints(); 
        //     pcl::toROSMsg(stable_points, laserCloudTemp);
        //     laserCloudTemp.header.stamp = cloud_header_.stamp;
        //     laserCloudTemp.header.frame_id = "base_link";
        //     stable_pointcloud_publisher_.publish(laserCloudTemp);
        // }

        // if (unstable_pointcloud_publisher_.getNumSubscribers() != 0) {
        //     pcl::PointCloud<UsedPointT> const& unstable_points = 
        //         segmentation_->GetUnStablePoints(); 
        //     pcl::toROSMsg(unstable_points, laserCloudTemp);
        //     laserCloudTemp.header.stamp = cloud_header_.stamp;
        //     laserCloudTemp.header.frame_id = "base_link";
        //     unstable_pointcloud_publisher_.publish(laserCloudTemp);
        // }

        // if (outlier_pointcloud_publisher_.getNumSubscribers() != 0) {
        //     pcl::PointCloud<UsedPointT> const& outlier_points = 
        //         segmentation_->GetOutlierPoints(); 
        //     pcl::toROSMsg(outlier_points, laserCloudTemp);
        //     laserCloudTemp.header.stamp = cloud_header_.stamp;
        //     laserCloudTemp.header.frame_id = "base_link";
        //     outlier_pointcloud_publisher_.publish(laserCloudTemp);
        // }

        // if (edge_feature_publisher_.getNumSubscribers() != 0) {
        //     CloudContainer<UsedPointT> const& processed_points = processed_points_.back();
        //     if (processed_points.pointcloud_data_.find("loam_edge") != processed_points.pointcloud_data_.end()) {
        //         pcl::toROSMsg(*processed_points.pointcloud_data_.at("loam_edge"), laserCloudTemp);
        //         laserCloudTemp.header.stamp = cloud_header_.stamp;
        //         laserCloudTemp.header.frame_id = "base_link";
        //         edge_feature_publisher_.publish(laserCloudTemp);
        //     }
        // }

        // if (surf_feature_publisher_.getNumSubscribers() != 0) {
        //     CloudContainer<UsedPointT> const& processed_points = processed_points_.back();
        //     if (processed_points.pointcloud_data_.find("loam_surf") != processed_points.pointcloud_data_.end()) {
        //         pcl::toROSMsg(*processed_points.pointcloud_data_.at("loam_surf"), laserCloudTemp);
        //         laserCloudTemp.header.stamp = cloud_header_.stamp;
        //         laserCloudTemp.header.frame_id = "base_link";
        //         surf_feature_publisher_.publish(laserCloudTemp);
        //     }
        // }
        if (surf_feature_publisher_.getNumSubscribers() != 0) {
            CloudContainer<UsedPointT> const& processed_points = processed_points_.back();
            if (processed_points.pointcloud_data_.find("bad_point") != processed_points.pointcloud_data_.end()) {
                pcl::toROSMsg(*processed_points.pointcloud_data_.at("bad_point"), laserCloudTemp);
                laserCloudTemp.header.stamp = cloud_header_.stamp;
                laserCloudTemp.header.frame_id = "base_link";
                surf_feature_publisher_.publish(laserCloudTemp);
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void publishLocalMapCloud(ros::Time const& stamp) {
        LidarTracker<UsedPointT>::LocalMapContainer local_maps = lidar_trackers_->GetLocalMap();
        for (auto const& iter : local_maps) {
            if (iter.first == "loam_edge") {
                // publishCloud( &pubLocalMapEdge[id],    // 发布该点云的话题 
                //                                 iter->second,   // 边缘特征   
                //                                 stamp, odom_frame);     
            } else if (iter.first == "loam_surf") {
                // publishCloud( &pubLocalMapSurf[id],    // 发布该点云的话题 
                //                                 iter->second,   // 点云数据   
                //                                 stamp, odom_frame);     
            } else if (iter.first == "filtered") {     // 滤波后的 
                publishCloud<UsedPointT>( &local_map_filtered_publisher_,    // 发布该点云的话题 
                                                iter.second,   // 点云数据   
                                                stamp, "odom");     
            }
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void pubMarker() {
        // if (markers_publisher_.getNumSubscribers()) {
            std::vector<PointCloudSegmentation<UsedPointT>::ClusterInfo> const& cluster_info 
                = segmentation_->GetClusterInfo();
            // std::vector<PointCloudRangeImageSegmentation<UsedPointT>::ClusterInfo> const& cluster_info 
            //     = segmentation_->GetClusterInfo();
            
            int size = cluster_info.size(); 

            visualization_msgs::MarkerArray markers;
            markers.markers.resize(size);
            for (int i = 0; i < size; i++) {
                // sphere
                visualization_msgs::Marker& sphere_marker = markers.markers[i];
                sphere_marker.header.frame_id = "base_link";
                sphere_marker.header.stamp = cloud_header_.stamp;
                sphere_marker.ns = "cluster";
                sphere_marker.id = i;
                sphere_marker.type = visualization_msgs::Marker::SPHERE;

                sphere_marker.pose.position.x = cluster_info[i].centre_.x();
                sphere_marker.pose.position.y = cluster_info[i].centre_.y();
                sphere_marker.pose.position.z = cluster_info[i].centre_.z();

                sphere_marker.pose.orientation.w = 1.0;
                sphere_marker.scale.x = sphere_marker.scale.y = 
                    sphere_marker.scale.z = 2 * cluster_info[i].direction_max_value_[2];
                //sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = 1;
                sphere_marker.color.r = 1.0;
                sphere_marker.color.a = 0.3;
            }
            markers_publisher_.publish(markers);
        // }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief: 激光去除畸变->获取激光匹配预测位姿->激光
     * @details: 
     * 激光+IMU融合(外参已知)：
     *      
     * 激光 + odom:
     *      1、初始化： 外参在线标定(平地，慢速，绕8字)
     *      2、odom给匹配初值 + 去畸变 (平地)
     * 激光 + odom&imu:
     *      1、初始化 
     *      2、odom+imu给匹配初值 + 去畸变 
     *      3、激光里程计 、odom + imu 滑动窗口优化
     *     
     */        
    void Estimator() {
        while(1) {
            std::unique_lock<std::mutex> laser_lock(laser_mt_);
            if (processed_points_.size() >= 2) {
                CloudContainer<UsedPointT> const& feature_data = processed_points_.front(); 
                Eigen::Isometry3d delta_pose_ = Eigen::Isometry3d::Identity();    // 运动增量   
                TicToc tt;
                lidar_trackers_->Solve(feature_data, delta_pose_);   
                tt.toc("lidar tracker "); 
                pose_ = pose_ * delta_pose_; // 当前帧的绝对运动   
                Eigen::Vector3f p = pose_.translation().cast<float>();
                Eigen::Quaternionf quat(pose_.rotation().cast<float>());
                quat.normalize();
                ros::Time stamp = ros::Time(feature_data.time_stamp_);  
                // 发布tf 
                PublishTF(p, quat, stamp, "odom", "lidar"); 
                // 发布localmap
                if (lidar_trackers_->HasUpdataLocalMap()) {
                    publishLocalMapCloud(stamp);
                }
                sensor_msgs::PointCloud2 laserCloudTemp;
                if (preprocessed_pointcloud_publisher_.getNumSubscribers() != 0) {
                    pcl::toROSMsg(*feature_data.pointcloud_data_.at("filtered"), laserCloudTemp);
                    laserCloudTemp.header.stamp = stamp;
                    laserCloudTemp.header.frame_id = "lidar";
                    preprocessed_pointcloud_publisher_.publish(laserCloudTemp);
                }
                processed_points_.pop_front(); 
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

private:
    using LidarTrackerPtr = std::unique_ptr<LidarTracker<UsedPointT>>;  
    enum LidarType {Livox = 0, Velodyne, Ouster};
    struct Option {
        LidarType lidar_type_; 
        std::string param_path_; 
        int num_scans_;   // 激光的线数 
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
    ros::Subscriber imu_subscriber_;                             // IMU
    ros::Subscriber gnss_subscriber_;                           // gnss   
    ros::Publisher preprocessed_pointcloud_publisher_;
    ros::Publisher stable_pointcloud_publisher_;  
    ros::Publisher unstable_pointcloud_publisher_;  
    ros::Publisher ground_pointcloud_publisher_;
    ros::Publisher nonground_pointcloud_publisher_;
    ros::Publisher outlier_pointcloud_publisher_;
    ros::Publisher edge_feature_publisher_;
    ros::Publisher surf_feature_publisher_;
    ros::Publisher local_map_filtered_publisher_;
    ros::Publisher markers_publisher_; // 可视化

    std_msgs::Header cloud_header_;

    std::string laser_frame = ""; 
    std::mutex laser_mt_;  

    std::deque<LidarData<UsedPointT>> lidar_buf_;
    std::queue<ImuDataPtr> imu_buf_;  
    std::queue<GnssDataPtr> gnss_buf_;    
    std::deque<CloudContainer<UsedPointT>> processed_points_;
    std::unique_ptr<LidarPreProcess<UsedPointT>> preprocess_;  
    LidarTrackerPtr lidar_trackers_;  
    //std::unique_ptr<PointCloudRangeImageSegmentation<UsedPointT>> segmentation_;  
    std::unique_ptr<PointCloudSegmentation<UsedPointT>> segmentation_;  
    std::unique_ptr<LOAMFeatureExtractor<UsedPointT, UsedPointT>> feature_extractor_;  
    pcl::PointCloud<UsedPointT>::ConstPtr preprocessed_cloud_; 
    Eigen::Isometry3d pose_;    // 运动
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "faster_loam");
    LIGORosNode node;
    std::thread estimator(&LIGORosNode::Estimator, &node); 
    ros::spin(); 
    return (0);
}







