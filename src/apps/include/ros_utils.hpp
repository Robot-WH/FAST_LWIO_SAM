
#ifndef _ROS_UTILS_HPP
#define _ROS_UTILS_HPP

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>
#include <condition_variable>

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>

#include <eigen3/Eigen/Dense>
#include <opencv/cv.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h> 
#include <pcl_conversions/pcl_conversions.h>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include "tool/file_manager.hpp"
#include "tic_toc.h"

#include <ros/ros.h>

#include <tf/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float64MultiArray.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <geometry_msgs/TransformStamped.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>
#include <livox_ros_driver/CustomMsg.h>

using namespace std;

using PointInType = pcl::PointXYZI;
using PointFeatureT = pcl::PointXYZI;

// velodyne 激光点云格式
struct EIGEN_ALIGN16 velodynePoint {
    PCL_ADD_POINT4D;
    float intensity;
    float time;
    std::uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
POINT_CLOUD_REGISTER_POINT_STRUCT(velodynePoint,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
    (float, time, time)(std::uint16_t, ring, ring));
// ouster 激光点云格式
struct EIGEN_ALIGN16 ousterPoint {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t ambient;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ousterPoint,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)(std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring)(std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range));


// ros 读取参数   
template <typename T>
static T RosReadParam(ros::NodeHandle &n, std::string name) {
    T ans;
    if (n.getParam(name, ans)) {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    } else {
        ROS_ERROR_STREAM("Failed to load " << name);
    }
    return ans;
}

template <typename _T>
static sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, 
        typename pcl::PointCloud<_T>::ConstPtr const& thisCloud, ros::Time const& thisStamp, 
        std::string const& thisFrame) {
    //thisCloud.width() = 0; 
    // std::cout<<"cloud.points.size (): "<<thisCloud->size()<<", cloud.width * cloud.height: "
    // <<thisCloud->width * thisCloud->height<<std::endl;
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    // std::cout<<"toROSMsg OK"<<std::endl;
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0) {
        thisPub->publish(tempCloud);
    }
    return tempCloud;
}

static void PublishOdometry(const ros::Time& stamp, const Eigen::Matrix4f& pose) {
    // // publish the transform                发布当前帧里程计到/odom话题                 
    // nav_msgs::Odometry odom;                            
    // odom.header.stamp = stamp;
    // odom.header.frame_id = odom_frame_id;       // odom坐标    /odom
    // odom.child_frame_id = lidar_frame_id;        // /lidar_odom
    // odom.pose.pose.position.x = pose(0, 3);
    // odom.pose.pose.position.y = pose(1, 3);
    // odom.pose.pose.position.z = pose(2, 3);
    
    // // 旋转矩阵 -> 四元数
    // Eigen::Quaternionf quat(pose.block<3, 3>(0, 0));
    // quat.normalize();
    // // 构造四元数   ROS信息
    // geometry_msgs::Quaternion odom_quat;
    // odom_quat.w = quat.w();
    // odom_quat.x = quat.x();
    // odom_quat.y = quat.y();
    // odom_quat.z = quat.z();
    // odom.pose.pose.orientation = odom_quat;  
    // odom_pub.publish(odom);
    // // 发布轨迹  
    // geometry_msgs::PoseStamped laserPose;    
    // laserPose.header = odom.header;
    // laserPose.pose = odom.pose.pose;                // 和laserOdometry的pose相同  
    // laserPath.header.stamp = odom.header.stamp;
    // laserPath.poses.push_back(laserPose);
    // laserPath.header.frame_id = odom_frame_id;      // odom坐标     /odom
    // pubLaserPath.publish(laserPath);

}

static void PublishTF(Eigen::Vector3f const& p, Eigen::Quaternionf const& quat, ros::Time const& stamp,
                                                string const& origin_id, string const& frame_id) {
    // 发布TF
	static tf::TransformBroadcaster br;
	tf::Transform transform;
    tf::Quaternion q;
	transform.setOrigin(tf::Vector3(p[0], p[1], p[2]));
    q.setW(quat.w());                               
	q.setX(quat.x());
	q.setY(quat.y());
	q.setZ(quat.z());    

	transform.setRotation(q);
	br.sendTransform(tf::StampedTransform(transform, stamp, origin_id, frame_id));
}

static void PubVisualizedMarkers(ros::NodeHandle &n, string const& topic_name, 
                                                                        visualization_msgs::MarkerArray const& markers) {
    static ros::Publisher markers_pub = 
        n.advertise<visualization_msgs::MarkerArray>(topic_name, 10);        // 可视化
    // 可视化     
    if(markers_pub.getNumSubscribers()) 
    {
        markers_pub.publish(markers);
    }  
}

template<typename T>
static double ROS_TIME(T msg) {
    return msg->header.stamp.toSec();
}

template<typename T>
static void imuAngular2rosAngular(sensor_msgs::Imu *thisImuMsg, T *angular_x, T *angular_y, T *angular_z) {
    *angular_x = thisImuMsg->angular_velocity.x;
    *angular_y = thisImuMsg->angular_velocity.y;
    *angular_z = thisImuMsg->angular_velocity.z;
}

template<typename T>
static void imuAccel2rosAccel(sensor_msgs::Imu *thisImuMsg, T *acc_x, T *acc_y, T *acc_z) {
    *acc_x = thisImuMsg->linear_acceleration.x;
    *acc_y = thisImuMsg->linear_acceleration.y;
    *acc_z = thisImuMsg->linear_acceleration.z;
}

template<typename T>
static void imuRPY2rosRPY(sensor_msgs::Imu *thisImuMsg, T *rosRoll, T *rosPitch, T *rosYaw) {
    double imuRoll, imuPitch, imuYaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(thisImuMsg->orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

    *rosRoll = imuRoll;
    *rosPitch = imuPitch;
    *rosYaw = imuYaw;
}

/**
 * @brief convert Eigen::Matrix to geometry_msgs::TransformStamped  Eigen转换为tf msg
 * @param stamp            timestamp
 * @param pose             Eigen::Matrix to be converted
 * @param frame_id         tf frame_id
 * @param child_frame_id   tf child frame_id
 * @return converted TransformStamped
 */
static geometry_msgs::TransformStamped matrix2transform(const ros::Time& stamp, 
                                                                                                                                const Eigen::Matrix4f& pose, 
                                                                                                                                const std::string& frame_id, 
                                                                                                                                const std::string& child_frame_id) {
  // 旋转矩阵 -> 四元数
  Eigen::Quaternionf quat(pose.block<3, 3>(0, 0));
  // 四元数单位化
  quat.normalize();
  // 构造四元数   ROS信息
  geometry_msgs::Quaternion odom_quat;
  odom_quat.w = quat.w();
  odom_quat.x = quat.x();
  odom_quat.y = quat.y();
  odom_quat.z = quat.z();
  
  geometry_msgs::TransformStamped odom_trans;
  odom_trans.header.stamp = stamp;
  // 该tf关系表示 从 frame_id-> child_frame_id
  odom_trans.header.frame_id = frame_id;       
  odom_trans.child_frame_id = child_frame_id;

  odom_trans.transform.translation.x = pose(0, 3);
  odom_trans.transform.translation.y = pose(1, 3);
  odom_trans.transform.translation.z = pose(2, 3);
  odom_trans.transform.rotation = odom_quat;

  return odom_trans;
}
// 输入: 位姿的ROS Msg
// 输出: Eigen变换矩阵
static Eigen::Isometry3d odom2isometry(const nav_msgs::OdometryConstPtr& odom_msg) {
  const auto& orientation = odom_msg->pose.pose.orientation;  
  const auto& position = odom_msg->pose.pose.position;
  // ROS   四元数转Eigen
  Eigen::Quaterniond quat;
  quat.w() = orientation.w;
  quat.x() = orientation.x;
  quat.y() = orientation.y;
  quat.z() = orientation.z;

  Eigen::Isometry3d isometry = Eigen::Isometry3d::Identity();
  // linear是获取旋转矩阵
  isometry.linear() = quat.toRotationMatrix();
  // 赋值平移
  isometry.translation() = Eigen::Vector3d(position.x, position.y, position.z);
  return isometry;
}    

#endif // ROS_UTILS_HPP
