/*
 * @Copyright(C): Your Company
 * @FileName: 文件名
 * @Author: 作者
 * @Version: 版本
 * @Date: 2022-04-28 23:22:00
 * @Description: 
 * @Others: 
 */

#pragma once 
#define PCL_NO_PRECOMPILE
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

/**
 * 6D位姿点云结构定义
*/
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D     
    PCL_ADD_INTENSITY;  
    float roll;         
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   
} EIGEN_ALIGN16;                    

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                (float, x, x) (float, y, y)
                                (float, z, z) (float, intensity, intensity)
                                (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                (double, time, time));

typedef PointXYZIRPYT  PointTypePose;

/**
 * XYZI + ring + time +range 点云结构
*/
struct PointXYZIRTD
{
    PCL_ADD_POINT4D     // 位置
    PCL_ADD_INTENSITY;  // 激光点反射强度，也可以存点的索引
    int16_t ring = -1;      // 扫描线
    float time = -1;         // 时间戳，记录相对于当前帧第一个激光点的时差，第一个点time=0
    float range = 0;   // 深度 
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;        // 内存16字节对齐，EIGEN SSE优化要求

// 注册为PCL点云格式
POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRTD,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (int16_t, ring, ring) (float, time, time) (float, range, range)
);
