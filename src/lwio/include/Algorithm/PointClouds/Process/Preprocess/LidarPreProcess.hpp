#pragma once

#include <yaml-cpp/yaml.h>
#include <pcl/filters/filter.h>
#include "Sensor/lidar_data_type.h"
#include "Common/pcl_type.h"
#include "../Filter/voxel_grid.hpp"
#include "../Filter/outlier_removal.hpp"

namespace Slam3D {
namespace Algorithm {
namespace {

using namespace Filter;  
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
            typename VoxelGridFilter<_PointT>::Option voxelGridFilter_option_;
            typename OutlierRemovalFilter<_PointT>::Option outlierRemovalFilter_option_; 
        };
    public:
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        LidarPreProcess() {}
        LidarPreProcess(Option const& option) 
        : option_(option) {
            downsampling_filter_.Reset(option_.voxelGridFilter_option_);
            outlier_filter_.Reset(option_.outlierRemovalFilter_option_);  
        }
        // 直接读取配置文件 初始化参数 
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
        virtual void Process(LidarData<_PointT>& data) {
            // 降采样
            if (is_begin_) {
                is_begin_ = false;
                if (option_.first_ignore_) {
                    return;
                }
            }
            if (option_.downsampling_enable_) {
                data.point_cloud = downsampling_filter_.Filter(data.point_cloud); 
            }
            // 离群点滤波
            if (option_.outlierRemoval_enable_) {
                data.point_cloud = outlier_filter_.Filter(data.point_cloud);
            }
        }
    protected:
        Option option_;  
        bool is_begin_ = true;  
        VoxelGridFilter<_PointT> downsampling_filter_;  
        OutlierRemovalFilter<_PointT> outlier_filter_;  
}; // class 
}
} // namespace Algorithm 
} // namespace Slam3D

