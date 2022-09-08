/*
 * @Copyright(C): 
 * @Author: lwh
 * @Description:  旋转激光雷达的预处理
 */

#pragma once

#include <pcl/filters/filter.h>
#include <glog/logging.h>
#include <yaml-cpp/yaml.h>
#include "Sensor/lidar_data_type.h"
#include "Common/pcl_type.h"
#include "LidarPreProcess.hpp"
#include "../LidarModel/RotatingLidarModel.hpp"

namespace Slam3D {
namespace Algorithm {
namespace {
using namespace Filter;  
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: 机械旋转雷达的预处理 
 * @details:  装饰模式 - 在滤波的基础上扩展求解每个点的时间戳的功能
 * @param _PointT 要有距离和时间通道
 */    
template<typename _PointT>
class RotaryLidarPreProcess : public LidarPreProcess<_PointT> {
    public:
        struct Option {
            float lidar_freq_ = 10;    // 默认 10 HZ 
            typename RotatingLidarModel<_PointT>::Option model_option_; 
            typename LidarPreProcess<_PointT>::Option filter_option_; 
            bool use_lidar_model_ = false;   // 是否计算雷达分线束模型
        };
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        RotaryLidarPreProcess(Option const& option) : option_(option), 
            common_preprocess_(option_.filter_option_), lidar_model_(option_.model_option_) {
        }
        // 直接读取配置文件 初始化参数 
        RotaryLidarPreProcess(const std::string&  path) {
            YAML::Node yaml = YAML::LoadFile(path);
            option_.filter_option_.downsampling_enable_ = yaml["preprocess"]["downsampling_enable"].as<bool>();
            option_.filter_option_.outlierRemoval_enable_ = yaml["preprocess"]["outlierRemoval_enable"].as<bool>();
            // 读取降采样器的参数 
            if (option_.filter_option_.downsampling_enable_) {
                LOG(INFO) << "set downsampling param!";
                std::string mode = yaml["preprocess"]["downsampling"]["mode"].as<std::string>();
                option_.filter_option_.voxelGridFilter_option_.mode_ = mode; 
                LOG(INFO) << "downsampling mode:" << mode;
               if (mode == "VoxelGrid") {
                   option_.filter_option_.voxelGridFilter_option_.voxel_grid_option_.resolution_ = 
                        yaml["preprocess"]["downsampling"]["VoxelGrid"]["resolution"].as<float>();
                    LOG(INFO) <<"VoxelGrid resolution:"<<option_.filter_option_.voxelGridFilter_option_.voxel_grid_option_.resolution_;
               } else if (mode == "ApproximateVoxelGrid") {
               }
            }
            if (option_.filter_option_.outlierRemoval_enable_) {
                LOG(INFO) << "set outlierRemoval param!";
            }
            common_preprocess_ = LidarPreProcess<_PointT>(option_.filter_option_); 
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 
         * @param[out] laser_cloud_in 
         */        
        void Process(LidarData<_PointT> &lidar_data) override {
            // 使用通用预处理器 完成滤波
            common_preprocess_.Process(lidar_data); 
            // 检查是否有时间戳信息
            static int check_time_channel = -1;
            if (check_time_channel == -1) {
                if (lidar_data.point_cloud->points[0].time == -1) {
                    LOG(INFO) << "激光点没有时间戳信息"; 
                    check_time_channel = 1;
                } else {
                    LOG(INFO) << "激光点有时间戳信息"; 
                    check_time_channel = 0; 
                }
            }
            if (check_time_channel) { // 没有时间戳信息则计算时间戳  
                calcPointTimestamp(*lidar_data.point_cloud);
            }
            // 如果需要计算旋转激光雷达的分线束模型 
            // loam 特征提取时要用到
            // 原始点云中的点排列可能是无序的，因此需要将点云重排列为有序的状态
            if (option_.use_lidar_model_) {
                calcLidarModel(lidar_data); 
            }
        }

    protected:
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 求解一帧激光起始和终止点角度 
         */    
        void findStartEndAngle(pcl::PointCloud<_PointT> const& laser_cloud_in, 
                float &start_ori, float &end_ori) {
            start_ori = -atan2(laser_cloud_in.points[0].y, laser_cloud_in.points[0].x);
            end_ori = -atan2(laser_cloud_in.points[laser_cloud_in.points.size() - 1].y,
                            laser_cloud_in.points[laser_cloud_in.points.size() - 1].x) + 2 * M_PI;
            if (end_ori - start_ori > 3 * M_PI) {
                end_ori -= 2 * M_PI;
            } else if (end_ori - start_ori < M_PI) {
                end_ori += 2 * M_PI;
            }
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void calcPointTimestamp(pcl::PointCloud<_PointT> &laser_cloud) {
            float start_ori, end_ori;
            findStartEndAngle(laser_cloud, start_ori, end_ori); // 计算起始点和终点的角度  
            bool half_passed = false;
            for (size_t i = 0; i < laser_cloud.size(); i++) {
                float ori = -atan2(laser_cloud.points[i].y, laser_cloud.points[i].x);
                if (!half_passed) {
                    if (ori < start_ori - M_PI / 2) {
                        ori += 2 * M_PI;
                    }
                    else if (ori > start_ori + M_PI * 3 / 2) {
                        ori -= 2 * M_PI;
                    }
                    if (ori - start_ori > M_PI) {
                        half_passed = true;
                    }
                } else {
                    ori += 2 * M_PI;
                    if (ori < end_ori - M_PI * 3 / 2) {
                        ori += 2 * M_PI;
                    }
                    else if (ori > end_ori + M_PI / 2) {
                        ori -= 2 * M_PI;
                    }
                }
                float rel_time = (ori - start_ori) / ((end_ori - start_ori) * option_.lidar_freq_);   // unit: s 
                laser_cloud.points[i].time = rel_time;
            }
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 计算每个点的ring,col信息，并将点云进行整理，按照scan顺序进行排列
         * @details: 
         * @param {LidarData<_PointT>} &lidar_data
         * @return {*}
         */        
        void calcLidarModel(LidarData<_PointT> &lidar_data) {
            // 激光内参自标定
            static bool calib_success = false;
            if (!calib_success) {
                if (lidar_model_.LidarIntrinsicCalibrate(*lidar_data.point_cloud)) {
                    calib_success = true;
                } else {
                    return; 
                }
            }
            // 获取模型的size
            //float const& horizon_resolution = lidar_model_.GetModelHorizonPointResolution(); 
            uint16_t const& SCAN_NUM = lidar_model_.GetModelHorizonPointNum();
            uint16_t const& LINE_NUM = lidar_model_.GetModelLineNum();
            pcl::PointCloud<_PointT>  orderly_points;
            orderly_points.resize(SCAN_NUM * LINE_NUM); 
            std::vector<int>  point_exist_flag;
            point_exist_flag.resize(SCAN_NUM * LINE_NUM, 0);
            // std::vector<float> col_angle;
            // col_angle.resize(SCAN_NUM, 0);

            for (uint32_t i = 0; i < lidar_data.point_cloud->size(); i++) {
                int16_t  ring = lidar_model_.GetRingID(lidar_data.point_cloud->points[i]);
                if (ring < 0) {
                    continue;
                }
                // 采用四舍五入的方式计算  有时相邻的点会计算出相同的列坐标
                std::pair<uint16_t, float> col_data = lidar_model_.GetColumn(lidar_data.point_cloud->points[i]);
                // if (col_angle[col_data.first] == 0) {
                //     col_angle[col_data.first] = col_data.second; 
                //     // if (ring == 0) {
                //     //     LOG(INFO) <<"col:"<<col_data.first<<",angle:"<<col_angle[col_data.first];
                //     // }
                // } else {   
                //                                                 if (ring != 0) {
                        
                //         continue; 
                //     }
                //     LOG(INFO) <<"col:"<<col_data.first<<",before angle:"<<col_angle[col_data.first]<<", curr:"<<col_data.second;
                //     if (round(fabs(col_data.second - col_angle[col_data.first]) / horizon_resolution) == 1) {
                //         //LOG(INFO) <<"col_data.second:"<<col_data.second<<",col_angle[col_data.first]:"<<col_angle[col_data.first];
                //         // if (col_angle[col_data.first] > col_data.second) {
                //         //     LOG(INFO) << " > col_angle[col_data.first + 1]:"<<col_angle[col_data.first + 1];
                //         //     LOG(INFO) << "col_angle[col_data.first]:"<<col_angle[col_data.first];
                //         //     LOG(INFO) << "col_data.second:"<<col_data.second;
                //         //     col_angle[col_data.first + 1] = col_angle[col_data.first];
                //         //     col_angle[col_data.first] = col_data.second;
                //         // } else {
                //         //     LOG(INFO) << "< col_angle[col_data.first + 1]:"<<col_angle[col_data.first + 1];
                //         //     LOG(INFO) << "col_angle[col_data.first]:"<<col_angle[col_data.first];
                //         //     LOG(INFO) << "col_data.second:"<<col_data.second;
                //         //     col_angle[col_data.first + 1] = col_data.second;
                //         //     col_data.first += 1; 
                //         // }
                //         if (col_data.second < col_angle[col_data.first]) {
                //             if (col_data.first == 0) {
                //                 col_data.first = SCAN_NUM - 1;
                //             } else {
                //                 col_data.first -= 1; 
                //             }
                //         } else {
                //             if (col_data.first == SCAN_NUM - 1) {
                //                 col_data.first = 0; 
                //             } else {
                //                 col_data.first += 1; 
                //             }
                //         }
                //         if (!col_angle[col_data.first]) {
                //             col_angle[col_data.first] = col_data.second; 
                //         }
                //         n++;
                //                                                                             if (ring == 0) {
                //             LOG(INFO) <<"after adjust  col:"<<col_data.first<<",before angle:"<<col_angle[col_data.first];
                //         }
                //     }
                // }
                lidar_data.point_cloud->points[i].ring = ring;
                if (point_exist_flag[ring * SCAN_NUM + col_data.first] == 0) {
                    orderly_points.points[ring * SCAN_NUM + col_data.first] = lidar_data.point_cloud->points[i]; 
                }
                point_exist_flag[ring * SCAN_NUM + col_data.first]++;  
            }
            lidar_data.point_col_idx_.clear(); 
            lidar_data.point_col_idx_.reserve(SCAN_NUM * LINE_NUM); 
            lidar_data.scan_start_idx_.clear();
            lidar_data.scan_start_idx_.resize(LINE_NUM, -1);   
            uint32_t count = 0; 
            int k = 0;
            lidar_data.point_cloud->clear(); 
            for (uint16_t i = 0; i < LINE_NUM; i++) {
                for (uint16_t j = 0; j < SCAN_NUM; j++) {
                    if (point_exist_flag[i * SCAN_NUM + j]) {
                        lidar_data.point_cloud->push_back(orderly_points.points[i * SCAN_NUM + j]); 
                        lidar_data.point_col_idx_.push_back(j);  // 记录 该点的水平列坐标 
                        if (lidar_data.scan_start_idx_[i] == -1) {    // 获取每一个line 起始位置的index  
                            lidar_data.scan_start_idx_[i] = count; 
                        }
                        count++; 
                    }
                    // if (point_exist_flag[i * SCAN_NUM + j] > 1) {
                    //     LOG(INFO) <<"i:"<<i<<", j:"<<j<<"point_exist_flag: "<<point_exist_flag[i * SCAN_NUM + j];
                    //     k++; 
                    // }
                }
            }
            // std::cout<<"repeate:"<<k<<std::endl;
            // std::cout<<"count:"<<count<<std::endl;
            // std::cout<<"after :"<<lidar_data.point_cloud->size()<<std::endl;
        }

    protected:
        Option option_;  
        LidarPreProcess<_PointT> common_preprocess_;   // 通用的预处理模块 
        RotatingLidarModel<_PointT> lidar_model_;  
        bool calc_timestamp_ = true;  // 默认要计算时间戳   
}; // class 
} // namespace 
}
}
