
#pragma once

#include <yaml-cpp/yaml.h>
#include <pcl/filters/filter.h>
#include "../../Sensor/lidar_data.h"
#include "SlamLib/PointCloud/Filter/voxel_grid.h"
#include "SlamLib/PointCloud/Filter/outlier_removal.h"
#include "SlamLib/tic_toc.hpp"

namespace lwio {

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: 激光雷达的通用预处理
 * @details:  滤波 (降采样、去离群点）
 */    
template<typename _PointT>
class LidarPreProcess {
public:
    struct Option {
        bool downsampling_enable_ = 1; 
        bool outlierRemoval_enable_ = 1; 
        bool first_ignore_ = 1; 
        SlamLib::pointcloud::FilterOption::VoxelGridFilterOption voxelGridFilter_option_;
        SlamLib::pointcloud::FilterOption::OutlierRemovalFilterOption outlierRemovalFilter_option_; 
    };
public:

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    LidarPreProcess() {}
    LidarPreProcess(Option const& option) 
    : option_(option) {
        downsampling_filter_.Reset(option_.voxelGridFilter_option_);
        outlier_filter_.Reset(option_.outlierRemovalFilter_option_);  
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    LidarPreProcess(const std::string&  path) {
        YAML::Node yaml = YAML::LoadFile(path);
        option_.downsampling_enable_ = yaml["preprocess"]["downsampling_enable"].as<bool>();
        option_.outlierRemoval_enable_ = yaml["preprocess"]["outlierRemoval_enable"].as<bool>();
        option_.first_ignore_ = yaml["preprocess"]["first_ignore_enable"].as<bool>();
        // 读取降采样器的参数 
        if (option_.downsampling_enable_) {
            LOG(INFO) << "set downsampling param!";
            std::string mode = yaml["preprocess"]["downsampling"]["mode"].as<std::string>();
            option_.voxelGridFilter_option_.mode_ = mode; 
            LOG(INFO) << "downsampling mode:" << mode;
            if (mode == "VoxelGrid") {
                option_.voxelGridFilter_option_.voxel_grid_option_.resolution_ = 
                    yaml["preprocess"]["downsampling"]["VoxelGrid"]["resolution"].as<float>();
                LOG(INFO) <<"VoxelGrid resolution:"<<option_.voxelGridFilter_option_.voxel_grid_option_.resolution_;
            } else if (mode == "ApproximateVoxelGrid") {
            }
        }
        if (option_.outlierRemoval_enable_) {
            LOG(INFO) << "set outlierRemoval param!";
        }
        downsampling_filter_.Reset(option_.voxelGridFilter_option_);
        outlier_filter_.Reset(option_.outlierRemovalFilter_option_);  
    }

    virtual ~LidarPreProcess() {}

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief: 点云的公共处理方法 - 比如livox雷达就会调用这个
     * @param  _InputType 任意类型的输入点云 
     * 
     */        
    virtual SlamLib::PCLPtr<_PointT> Process(const LidarData<_PointT>& data) {
        // 降采样
        if (is_begin_) {    // 如果是第一帧那么不需要降采样  
            is_begin_ = false;
            if (option_.first_ignore_) {
                return data.pointcloud_ptr_;
            }
        }
        SlamLib::PCLPtr<_PointT> filtered_points; 
        // std::cout << "before filter size: " <<  data.pointcloud_ptr_->size() << std::endl;
        // 降采样滤波 
        if (option_.downsampling_enable_) {
            filtered_points = downsampling_filter_.Filter(data.pointcloud_ptr_); 
        }
        // 离群点滤波
        if (option_.outlierRemoval_enable_) {
            filtered_points = outlier_filter_.Filter(filtered_points);
        }
        // std::cout << "after filter size: " <<  filtered_points->size() << std::endl;
        // 重新计算range   pcl的bug   滤波后range会变化 
        for (auto& p : *filtered_points) {
            p.range = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        }
        return filtered_points;  
    }
protected:
    Option option_;  
    bool is_begin_ = true;  
    SlamLib::pointcloud::VoxelGridFilter<_PointT> downsampling_filter_;  
    SlamLib::pointcloud::OutlierRemovalFilter<_PointT> outlier_filter_;  
}; // class 
} // namespace Slam3D

