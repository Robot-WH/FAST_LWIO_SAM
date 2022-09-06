/*
 * @Copyright(C): 
 * @FileName: 
 * @Author: lwh
 * @Description:  一种简单实现 lidar,imu,odom融合的前端里程计
 *                                  特点： 0、基于faster_loam 
 *                                                 1、imu-odom组合成3D里程计，用于去畸变以及预测
 *                                                 2、参考liosam构造因子图估计imu的bias
 *                                                 3、odom-lidar 外参标定 
 *                                                 
 */
#define PCL_NO_PRECOMPILE
#include "ros_utils.hpp"
#include "Estimator/dead_reckoning.hpp"
#include "Sensor/sensor.hpp"
#include "Sensor/lidar_data_type.h"
#include "Common/pcl_type.h"
#include "Algorithm/PointClouds/Process/Preprocess/RotaryLidarPreProcess.hpp"
#include "Algorithm/PointClouds/Process/Segmentation/PointCloudRangeImageSegmentation.hpp"
#include "Algorithm/PointClouds/Process/Segmentation/PointCloudSegmentation.hpp"

constexpr bool PARAM_ESTIMATE = true;  // 外参估计 
using namespace Slam3D; 
using namespace Algorithm; 
using PointT = PointXYZIRTD;

class FusionOdometryNode {
    public:
        struct Option {
            float min_laser_range_ = -1;
            float max_laser_range_ = -1;  
        };
        FusionOdometryNode() {
            // \033[1;32m，\033[0m 终端显示成绿色
            ROS_INFO_STREAM("\033[1;32m----> optimization fusion node started.\033[0m");
            imu_subscriber_ = node_handle_.subscribe(
                "imu", 2000, &FusionOdometryNode::ImuCallback, this, ros::TransportHints().tcpNoDelay());
            wheel_odom_subscriber_ = node_handle_.subscribe(
                "odom_scout", 2000, &FusionOdometryNode::OdomCallback, this, ros::TransportHints().tcpNoDelay());
            lidar_subscriber_ = node_handle_.subscribe(
                // "/rslidar_points" /points_raw   /left/velodyne_points
                "/rslidar_points", 5, &FusionOdometryNode::LidarCallback, this, ros::TransportHints().tcpNoDelay());
            odom_pub = node_handle_.advertise<nav_msgs::Odometry>(
                "dead_reckoning", 1, this);
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
            markers_publisher_ = node_handle_.advertise<visualization_msgs::MarkerArray>(
                "/cluster_markers", 10);        // 可视化
            // AccSigmaLimit_ = 0.002;
            Init();  
        }

        // 读参数 + 初始化
        void Init() {
            DeadReckon3D::Option option;
            option.use_odom_motion_ = false;  
            dead_reckon_ = DeadReckon3D(option); 

            RotaryLidarPreProcess<PointT>::Option preprocess_option; 
            preprocess_option.lidar_freq_ = 19.3;   
            preprocess_option.filter_option_.max_range_thresh_ = 100;  
            preprocess_.reset(new RotaryLidarPreProcess<PointT>(preprocess_option)); 

            // PointCloudRangeImageSegmentation<PointT>::Option seg_option; 
            // seg_option.horizon_angle_resolution_ = 0.386;
            // seg_option.segment_angle_thresh_ = 0.05;
            // segmentation_.reset(new PointCloudRangeImageSegmentation<PointT>(seg_option)); 
            PointCloudSegmentation<PointT>::Option seg_option; 
            seg_option.horizon_angle_resolution_ = 0.386;
            seg_option.segment_angle_thresh_ = 0.05;
            segmentation_.reset(new PointCloudSegmentation<PointT>(seg_option)); 
            // PointCloudSegmentation<PointT>::Option seg_option; 
            // segmentation_.reset(new PointCloudSegmentation<PointT>(seg_option)); 
        }

        // imu的回调函数
        void ImuCallback(const sensor_msgs::Imu::ConstPtr &imuMsg) {
            std::lock_guard<std::mutex> lock(imu_mt_);
            ImuData imu; 
            imu.timestamp_ = imuMsg->header.stamp.toSec();
            imu.acc_ = Eigen::Vector3d{imuMsg->linear_acceleration.x, 
                                                                    imuMsg->linear_acceleration.y,
                                                                    imuMsg->linear_acceleration.z};
            imu.gyro_ = Eigen::Vector3d{imuMsg->angular_velocity.x,
                                                                        imuMsg->angular_velocity.y,
                                                                        imuMsg->angular_velocity.z};        
            imu.rot_.w() = imuMsg->orientation.w;
            imu.rot_.x() = imuMsg->orientation.x;
            imu.rot_.y() = imuMsg->orientation.y;
            imu.rot_.z() = imuMsg->orientation.z;
            imu_buf_.push_back(imu);
           // if (imu_init_) {
                dead_reckon_.AddImuData(imu, acc_bias_, gyro_bias_);  
            // } else {
                // // 初始化
                // if (ImuInit(imu_buf_)) {
                //     imu_init_ = true;  
                // }
            //}
        }

        // // imu初始化 - 静止情况下，计算出角速度的bias 
        // bool ImuInit(std::deque<ImuData> &imu_data) {
        //     if (imu_data.size() < 100) {
        //         return false;
        //     }
        //     Eigen::Vector3d sigma_acc{0., 0., 0.};
        //     Eigen::Vector3d mean_acc{0., 0., 0.};
        //     Eigen::Vector3d mean_gyro{0., 0., 0.};
        //     uint16_t N = 0;
        //     // 计算均值
        //     for (auto const& imu : imu_data) {
        //         mean_gyro += (imu.angular_v_ - mean_gyro) / (N + 1);  
        //         mean_acc += (imu.acc_ - mean_acc) / (N + 1); 
        //         // cov_acc = cov_acc * (N / (N + 1)) + 
        //         //     (imu.acc_ - mean_acc).cwiseProduct(imu.acc_ - mean_acc) * N / ((N + 1) * (N + 1));
        //         N++; 
        //     }
        //     // 计算加速度方差   判定是否静止 
        //     for (auto const& imu : imu_data) {
        //         sigma_acc += (imu.acc_ - mean_acc).cwiseAbs2();
        //     }
        //     sigma_acc = (sigma_acc / imu_data.size()).cwiseSqrt(); 
        //     if (sigma_acc.norm() < AccSigmaLimit_) {
        //         yaw_angular_velocity_bias_ = mean_gyro[2]; 
        //         std::cout<<"imu yaw angular velocity bias init ok! is: "<<yaw_angular_velocity_bias_<<std::endl;
        //         return true;
        //     }
        //     imu_data.clear(); 
        //     return false; 
        // }

        // odom的回调函数
        void OdomCallback(const nav_msgs::Odometry::ConstPtr &odometryMsg) {
            std::lock_guard<std::mutex> lock(wheel_odom_mt_);
            OdomData odom;
            odom.timestamp_ = odometryMsg->header.stamp.toSec();
            odom.velocity_ = odometryMsg->twist.twist.linear.x;   // 线速度即为 x轴的速度  
            odom.yaw_angular_vel_ = odometryMsg->twist.twist.angular.z; // yaw的角速度
            // odom位姿     odom为2D，没有z的观测
            odom.pos_xy_.x() = odometryMsg->pose.pose.position.x;
            odom.pos_xy_.y() = odometryMsg->pose.pose.position.y;
            odom.orientation_.w() = odometryMsg->pose.pose.orientation.w;
            odom.orientation_.x() = odometryMsg->pose.pose.orientation.x;
            odom.orientation_.y() = odometryMsg->pose.pose.orientation.y;
            odom.orientation_.z() = odometryMsg->pose.pose.orientation.z;
            odom_buf_.push_back(odom);
            dead_reckon_.AddOdometryData(odom);   // 进行航迹推算
            DeadReckon3D::Pose odom_pose = dead_reckon_.GetDeadReckoningPose();
            // 发布航迹推算的结果
            nav_msgs::Odometry odom_msg;
            odom_msg.header = odometryMsg->header;
            odom_msg.pose.pose.position.x = odom_pose.x_;
            odom_msg.pose.pose.position.y = odom_pose.y_;
            odom_msg.pose.pose.position.z = odom_pose.z_;
            odom_msg.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(odom_pose.roll_, 
                odom_pose.pitch_,odom_pose.yaw_);
            odom_pub.publish(odom_msg); 
            // tf
            // static tf::TransformBroadcaster br;
            // tf::Transform transform;
            // tf::Quaternion q;
            // transform.setOrigin(tf::Vector3(odom_pose.x_, odom_pose.y_, 0));
            // q.setW(odom_pose.GetOrientation().w());                               
            // q.setX(odom_pose.GetOrientation().x());
            // q.setY(odom_pose.GetOrientation().y());
            // q.setZ(odom_pose.GetOrientation().z());    

            // transform.setRotation(q);
            // br.sendTransform(tf::StampedTransform(transform, odometryMsg->header.stamp, "odom", "base_link"));
        }

        // 激光的回调  
        void LidarCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg) {
            // 检查是否存在ring通道，注意static只检查一次
            if (ringFlag_ == -1) {
                ringFlag_ = 0;
                for (int i = 0; i < (int)cloud_msg->fields.size(); ++i) {
                    if (cloud_msg->fields[i].name == "ring") {
                        ringFlag_ = 1;
                        std::cout<<"有点云ring信息"<<std::endl;
                        break;
                    }
                }
                if (ringFlag_ == 0) {
                    std::cout<<"无点云ring信息, 请自行计算!"<<std::endl;
                }
            }
            // 检查是否存在time通道
            if (timeFlag_ == -1) {
                timeFlag_ = 0;
                for (auto &field : cloud_msg->fields) {
                    if (field.name == "time" || field.name == "t") {
                        timeFlag_ = 1;
                        std::cout<<"有点云point时间戳信息"<<std::endl;
                        break;
                    }
                }
                if (timeFlag_ == 0) {
                    std::cout<<"无点云point时间戳, 请自行计算!"<<std::endl;
                }
            }
            
            cloud_header_ = cloud_msg->header;
            LidarData<PointT> data;   
            data.timestamp = cloud_msg->header.stamp.toSec();
            pcl::fromROSMsg(*cloud_msg, *data.point_cloud);
            // 去除Nan
            std::vector<int> indices;
            pcl::removeNaNFromPointCloud<PointT>(*data.point_cloud, *data.point_cloud, indices);
            lidar_buf_.emplace_back(std::move(data));
            // 有2个以上的lidar数据才进行处理，确保其他传感器数据能进行覆盖
            if (lidar_buf_.size() >= 2) {
                LidarData<PointT>& data = lidar_buf_.front(); 
                // pcl::PointCloud<PointT> cloud_processed;
                // // 点云预处理(给出时间戳+滤波)
                // TicToc tt; 
                // preprocess_->Process(data.point_cloud, cloud_processed); 
                // // 去畸变
                // // 点云分割
                // segmentation_->Process(cloud_processed); 
                // double time = tt.toc("process: ");
                // static double avg_time = 0;
                // static int N = 0;
                // avg_time += (time - avg_time) / (N + 1);  
                // std::cout<<"avg time: "<<avg_time<<std::endl;
                // 特征提取 
                lidar_buf_.pop_front(); 
                // 点云发布
                publishCloud();
                pubMarker(); 
            }
        }

        void publishCloud() {
            sensor_msgs::PointCloud2 laserCloudTemp;
            if (nonground_pointcloud_publisher_.getNumSubscribers() != 0) {
                pcl::PointCloud<PointT> const& nonground_points = 
                    segmentation_->GetNonGroundPoints(); 
                pcl::toROSMsg(nonground_points, laserCloudTemp);
                laserCloudTemp.header.stamp = cloud_header_.stamp;
                laserCloudTemp.header.frame_id = "base_link";
                nonground_pointcloud_publisher_.publish(laserCloudTemp);
            }

            if (ground_pointcloud_publisher_.getNumSubscribers() != 0) {
                pcl::PointCloud<PointT> const& ground_points = 
                    segmentation_->GetGroundPoints(); 
                pcl::toROSMsg(ground_points, laserCloudTemp);
                laserCloudTemp.header.stamp = cloud_header_.stamp;
                laserCloudTemp.header.frame_id = "base_link";
                ground_pointcloud_publisher_.publish(laserCloudTemp);
            }

            if (stable_pointcloud_publisher_.getNumSubscribers() != 0) {
                pcl::PointCloud<PointT> const& stable_points = 
                    segmentation_->GetStablePoints(); 
                pcl::toROSMsg(stable_points, laserCloudTemp);
                laserCloudTemp.header.stamp = cloud_header_.stamp;
                laserCloudTemp.header.frame_id = "base_link";
                stable_pointcloud_publisher_.publish(laserCloudTemp);
            }

            if (unstable_pointcloud_publisher_.getNumSubscribers() != 0) {
                pcl::PointCloud<PointT> const& unstable_points = 
                    segmentation_->GetUnStablePoints(); 
                pcl::toROSMsg(unstable_points, laserCloudTemp);
                laserCloudTemp.header.stamp = cloud_header_.stamp;
                laserCloudTemp.header.frame_id = "base_link";
                unstable_pointcloud_publisher_.publish(laserCloudTemp);
            }

            if (outlier_pointcloud_publisher_.getNumSubscribers() != 0) {
                pcl::PointCloud<PointT> const& outlier_points = 
                    segmentation_->GetOutlierPoints(); 
                pcl::toROSMsg(outlier_points, laserCloudTemp);
                laserCloudTemp.header.stamp = cloud_header_.stamp;
                laserCloudTemp.header.frame_id = "base_link";
                outlier_pointcloud_publisher_.publish(laserCloudTemp);
            }
        }

        void pubMarker() {
            // if (markers_publisher_.getNumSubscribers()) {
                std::vector<PointCloudSegmentation<PointT>::ClusterInfo> const& cluster_info 
                    = segmentation_->GetClusterInfo();
                // std::vector<PointCloudRangeImageSegmentation<PointT>::ClusterInfo> const& cluster_info 
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
         *      3、激光里程计 、odom + imu 滑动窗口优化，
         *     
         */        
        void Estimator() {
            while(1) {
                std::unique_lock<std::mutex> laser_lock(laser_mt_);

                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
        }


    private:
        enum WorkStatus {
            INIT = 0,    // 初始化状态
            PARAM_OPT, // 参数优化阶段， 滑动窗口优化  
            ODOMETRY // 里程计运行, 基于滤波器，可以在线优化参数   也可不优化   
        };
        struct LaserInfo {
            std::vector<double> index_cos_; // 保存下来雷达各个角度的cos值
            std::vector<double> index_sin_; // 保存下来雷达各个角度的sin值
            float angle_increment_;  // 激光点角度增量  
            double time_increment_;  
        }laser_info_;

        WorkStatus status = INIT;   
        Option option_; 
        ros::NodeHandle node_handle_;  
        ros::Subscriber imu_subscriber_;
        ros::Subscriber wheel_odom_subscriber_;
        ros::Subscriber lidar_subscriber_;
        ros::Publisher odom_pub;
        ros::Publisher stable_pointcloud_publisher_;  
        ros::Publisher unstable_pointcloud_publisher_;  
        ros::Publisher ground_pointcloud_publisher_;
        ros::Publisher nonground_pointcloud_publisher_;
        ros::Publisher outlier_pointcloud_publisher_;
        ros::Publisher markers_publisher_; // 可视化

        std_msgs::Header cloud_header_;

        std::string laser_frame = ""; 

        std::mutex imu_mt_, wheel_odom_mt_, laser_mt_;  

        std::deque<OdomData> odom_buf_;
        std::deque<ImuData> imu_buf_;
        std::deque<LidarData<PointT>> lidar_buf_;
        /**
         * @todo 作为状态放到estimator对象中去
         */        
        Eigen::Vector3d acc_bias_{0., 0., 0.};
        Eigen::Vector3d gyro_bias_{0., 0., 0.};
        // 模块
        DeadReckon3D dead_reckon_; 
        std::unique_ptr<RotaryLidarPreProcess<PointT>> preprocess_;  
        //std::unique_ptr<PointCloudRangeImageSegmentation<PointT>> segmentation_;  
        std::unique_ptr<PointCloudSegmentation<PointT>> segmentation_;  
        bool imu_init_ = false; 
        int8_t ringFlag_ = -1; 
        int8_t timeFlag_ = -1; 
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "optimization_fusion_odometry_node");
    FusionOdometryNode node;
    std::thread estimator(&FusionOdometryNode::Estimator, &node); 
    ros::spin(); 
    return (0);
}







