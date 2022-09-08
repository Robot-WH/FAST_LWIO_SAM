/*
 * @Copyright(C): 
 * @Author: lwh
 * @Description: 参考loam的特征提取  
 * @Others:  ref: floam   
 */

#pragma once 

#include "Sensor/lidar_data_type.h"
#include "../Filter/voxel_grid.hpp"

namespace Slam3D {
namespace Algorithm {

using Slam3D::CloudContainer;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: loam的特征提取方法基类     后续的子类通过装饰模式对其进行扩展 
 * @details 对于多线雷达采用分层的方式提取特征 
 *                     适合于线束不高的雷达使用，例如 16线、32，64线等 
 */    
template<typename _InputPointT, typename _OutputFeatureT>
class LOAMFeatureExtractor {
    private:
        using InputPointCloud = pcl::PointCloud<_InputPointT>;  
        using FeaturePointCloud = pcl::PointCloud<_OutputFeatureT>;  
        using InputPointCloudPtr = typename pcl::PointCloud<_InputPointT>::Ptr; 
        using FeaturePointCloudPtr = typename pcl::PointCloud<_OutputFeatureT>::Ptr; 
        using InputPointCloudConstPtr = typename pcl::PointCloud<_InputPointT>::ConstPtr; 
    public:
        // 参数配置 
        struct Option {
            float lidar_freq_ = 10;   // 激光的频率   
            float horizon_angle_resolution_ = 0.2;   // 水平角分辨率 
            float edge_thresh_ = 0.1; 
            bool check_bad_point_ = true; 
        };

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        LOAMFeatureExtractor(Option option) : option_(option) {
            // 设置降采样滤波器
            // down_sampling_edge_.Reset("VoxelGrid", surf_voxel_grid_size);
            // down_sampling_surf_.Reset("VoxelGrid", 2 * surf_voxel_grid_size);
            rotation_angle_v_ =  option_.lidar_freq_ * 360;  // 角度 / s     
            horizon_angle_resolution_rad_ = option_.horizon_angle_resolution_ * M_PI / 180;                   
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 特征提取
         * @param data_in 输入的激光数据 
         * @param data_out 输出特征结果   
         * @return 
         */            
        void Extract(LidarData<_InputPointT> const& data_in, 
                CloudContainer<_OutputFeatureT> &data_out) {
            FeaturePointCloudPtr pc_out_edge(new FeaturePointCloud());    // 线特征
            FeaturePointCloudPtr pc_out_surf(new FeaturePointCloud());      // 面特征
            FeaturePointCloudPtr pc_bad_point(new FeaturePointCloud());      // 面特征
            std::vector<uint32_t> scan_index;
            // 首先给输入的多线激光数据分scan
            //splitScan(*data_in.point_cloud, scan_index);
            assert(data_in.scan_start_idx_.size() != 0);
            std::vector<int> disable_point;                                         // 标记被选择的点  
            disable_point.resize(data_in.point_cloud->points.size(), 0); 
            std::vector<int> is_edge_points;                                         // 标记被选择的点  
            is_edge_points.resize(data_in.point_cloud->points.size(), 0); 
            uint32_t scan_start_index, scan_end_index; 
            uint16_t total_points; 
            // 遍历雷达的每一个scan 分层提取特征
            for (uint16_t i = 0; i < data_in.scan_start_idx_.size(); i++) {
                scan_start_index = data_in.scan_start_idx_[i];
                // 当前ring层用于提取特征的点的总数 
                if (i == data_in.scan_start_idx_.size() - 1) {
                    scan_end_index = data_in.point_cloud->size() - 1;
                } else {
                    scan_end_index = data_in.scan_start_idx_[i + 1] - 1;
                }
                total_points = scan_end_index - scan_start_index - 9;    // scan_end_index - scan_start_index + 1 - 10 
                if (total_points < 6) continue;  
                // 将该scan中不应该提取edge特征的部分标记
                if (check_bad_points_) {
                    checkBadEdgePoint(*data_in.point_cloud, scan_start_index, scan_end_index, disable_point, pc_bad_point);  
                }
                uint16_t sector_length = (uint16_t)((total_points / 6) + 0.5);        // 四舍五入      
                // 每一个scan 分成6个sector 进行
                for(uint8_t k = 0; k < 6; k++) {
                    uint32_t sector_start_index = scan_start_index + 5 + sector_length * k;      // 该sector起始点在laserCloudScans[i]的序号 
                    uint32_t sector_end_index = sector_start_index + sector_length -1;  
                    if (k == 5) {
                        sector_end_index = scan_end_index - 5; 
                    }
                    // 保存这个sector的曲率信息 
                    std::vector<CurvatureInfo> cloud_curvature; 
                    cloud_curvature.reserve(total_points);  
                    // 计算这个sector 所有点的曲率     归一化处理
                    for(uint32_t j = sector_start_index; j <= sector_end_index; j++) {
                        // if (disable_point[j]) continue;  
                        double diffX = (data_in.point_cloud->points[j - 5].x + data_in.point_cloud->points[j - 4].x 
                                                    + data_in.point_cloud->points[j - 3].x + data_in.point_cloud->points[j - 2].x 
                                                    + data_in.point_cloud->points[j - 1].x - 10 * data_in.point_cloud->points[j].x 
                                                    + data_in.point_cloud->points[j + 1].x + data_in.point_cloud->points[j + 2].x
                                                    + data_in.point_cloud->points[j + 3].x + data_in.point_cloud->points[j + 4].x 
                                                    + data_in.point_cloud->points[j + 5].x);
                        double diffY = data_in.point_cloud->points[j - 5].y + data_in.point_cloud->points[j - 4].y 
                                                    + data_in.point_cloud->points[j - 3].y + data_in.point_cloud->points[j - 2].y 
                                                    + data_in.point_cloud->points[j - 1].y - 10 * data_in.point_cloud->points[j].y 
                                                    + data_in.point_cloud->points[j + 1].y + data_in.point_cloud->points[j + 2].y 
                                                    + data_in.point_cloud->points[j + 3].y + data_in.point_cloud->points[j + 4].y 
                                                    + data_in.point_cloud->points[j + 5].y;
                        double diffZ = data_in.point_cloud->points[j - 5].z + data_in.point_cloud->points[j - 4].z 
                                                    + data_in.point_cloud->points[j - 3].z + data_in.point_cloud->points[j - 2].z 
                                                    + data_in.point_cloud->points[j - 1].z - 10 * data_in.point_cloud->points[j].z 
                                                    + data_in.point_cloud->points[j + 1].z + data_in.point_cloud->points[j + 2].z 
                                                    + data_in.point_cloud->points[j + 3].z + data_in.point_cloud->points[j + 4].z 
                                                    + data_in.point_cloud->points[j + 5].z;
                        // 归一化 
                        // double curvature = (diffX * diffX + diffY * diffY + diffZ * diffZ) / 
                        //     (data_in.point_cloud->points[j].range * data_in.point_cloud->points[j].range); 
                        double curvature = diffX * diffX + diffY * diffY + diffZ * diffZ; 
                        cloud_curvature.emplace_back(j, curvature);
                    }
                    featureExtractionFromSector(data_in.point_cloud, cloud_curvature, disable_point, 
                                                                                    is_edge_points, pc_out_edge, pc_out_surf);
                }
            }    
            data_out.pointcloud_data_.insert(make_pair("loam_edge", std::move(pc_out_edge)));  
            data_out.pointcloud_data_.insert(make_pair("loam_surf", std::move(pc_out_surf)));  
            data_out.pointcloud_data_.insert(make_pair("bad_point", std::move(pc_bad_point)));
        } 

    protected:
        class  CurvatureInfo {
            public:
                int id;
                double value;  // 曲率值 
                CurvatureInfo(int const& id_in, double const& value_in) 
                : id(id_in), value(value_in) {}
        };

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 在一个scan的sector中提取特征
         * @param pc_in 该scan的全部点
         * @param cloud_curvature 该sector中点的曲率信息 
         * @return 
         */            
        void featureExtractionFromSector(InputPointCloudConstPtr const& pc_in, 
                                                                                std::vector<CurvatureInfo>& cloud_curvature, 
                                                                                std::vector<int> &disable_point,
                                                                                std::vector<int> &is_edge_points,
                                                                                FeaturePointCloudPtr& pc_out_edge, 
                                                                                FeaturePointCloudPtr& pc_out_surf) {
            std::sort(cloud_curvature.begin(), cloud_curvature.end(), 
                [](const CurvatureInfo & a, const CurvatureInfo & b) { 
                return a.value < b.value;  // 曲率从小到大 升序排序，   > 为降序
            });
            int PickedEdgeNum = 0;
            // 曲率从大到小遍历
            for (int i = cloud_curvature.size() - 1; i >= 0; i--) {
                int ind = cloud_curvature[i].id; 
                // 如果该点可以被选
                if (disable_point[ind] == 0) {
                    //  边缘特征的阈值    边缘特征的阈值大一点   这样提取的较为严谨  
                    if(cloud_curvature[i].value <= option_.edge_thresh_) {
                        break;
                    }
                    PickedEdgeNum++;
                    // 最多选择20个特征出来用 
                    if (PickedEdgeNum <= 20) {
                        pc_out_edge->push_back(pc_in->points[ind]);    
                        is_edge_points[ind] = 1;
                    } else {
                        break;
                    }
                    // 防止聚集    该点临近的5个点都不能提取边缘特征 
                    for(int k = 1; k <= 5; k++) {
                        int n = ind + k;
                        if (n >= disable_point.size()) {
                            break; 
                        }
                        disable_point[n] = 1;
                    }
                    //  判断前5个点是否聚集   
                    for(int k=-1;k>=-5;k--) {
                        int n = ind + k;
                        if (n < 0) {
                            continue; 
                        }  
                        disable_point[n] = 1;
                    }
                }
            }
            // 剩下的全部当成面特征    
            for (int i = 0; i <= (int)cloud_curvature.size() - 1; i++) {
                int ind = cloud_curvature[i].id; 
                // 不要考虑阈值  因为 由于靠近边缘点的原因，有些平面点曲率计算出来会很大, 这样会漏掉一些表面点
                // 只要不是边缘点  全部作为表面点  
                if (is_edge_points[ind] == 0) {
                    pc_out_surf->push_back(pc_in->points[ind]);
                }
            }
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 检测当前scan中不好的边缘点   提前标记出来  
         * @details:  edge特征的提取较为严格，需要满足一下条件：
         *                          1、整个采样范围内是连续的  2、edge点处的采样分辨率足够小( < 0.1m)
         *                       处理两个特殊情况 :
         *                          1、遮挡点   2、平行观测
         * @param pc_in 输入点云
         * @param scan_start_idx 当前scan帧起始序号 
         * @param scan_end_idx 当前scan帧最后的序号 
         * @param disable_point 标记 
         */            
        virtual void checkBadEdgePoint(InputPointCloud const& pc_in, 
                                                                                uint32_t const& scan_start_idx, 
                                                                                uint32_t const& scan_end_idx, 
                                                                                std::vector<int> &disable_point,
                                                                                FeaturePointCloudPtr &pc_bad_point) {                            
            int n = 0;                   
            for (int j = scan_start_idx + 5; j <= scan_end_idx - 5; j++) {
                if (disable_point[j]) continue; 
                // step1、检测不连续点
                // 检查角度 - 如果前后两个点的角度差别 >= 2倍原角分辨率 认为出现了跳变 
                double delta_angle = rotation_angle_v_ * (pc_in.points[j + 1].time - pc_in.points[j].time);
                if (delta_angle >= 1.5 * option_.horizon_angle_resolution_) {
                    // 先将该位置的前5个点进行设置
                    disable_point[j] = 1;
                    disable_point[j - 5] = 1;
                    disable_point[j - 4] = 1;
                    disable_point[j - 3] = 1;
                    disable_point[j - 2] = 1;
                    disable_point[j - 1] = 1;
                    pc_bad_point->push_back(pc_in.points[j]); 
                    pc_bad_point->push_back(pc_in.points[j - 1]); 
                    pc_bad_point->push_back(pc_in.points[j - 2]); 
                    pc_bad_point->push_back(pc_in.points[j - 3]); 
                    pc_bad_point->push_back(pc_in.points[j - 4]); 
                    pc_bad_point->push_back(pc_in.points[j - 5]); 
                    // 向后设置
                    uint8_t k = 5;
                    while (k > 0) {
                        k--; 
                        j++; 
                        if (j >= scan_end_idx - 5) break; 
                        disable_point[j] = 1;
                        pc_bad_point->push_back(pc_in.points[j]); 
                        if (rotation_angle_v_ * (pc_in.points[j + 1].time - pc_in.points[j].time) 
                                >= 1.5 * option_.horizon_angle_resolution_) {
                            k = 5; 
                        }
                    }
                    continue;  
                }
                // step2、检查稀疏性  太稀疏也不应该提取边缘特征
                if (pc_in.points[j].range * horizon_angle_resolution_rad_ > 0.1) {
                    disable_point[j] = 1;
                    continue;
                }
                // step3、检测遮挡  
                double curr_range = pc_in.points[j].range;  
                double next_range = pc_in.points[j + 1].range;  
                double tan_theta;
                if (curr_range < next_range) {
                    tan_theta = curr_range * horizon_angle_resolution_rad_ / (next_range - curr_range);
                } else {
                    tan_theta = next_range * horizon_angle_resolution_rad_ / (curr_range - next_range);
                }
                if (tan_theta <= 0.087) {   // 夹角小于 5度   则认为出现了遮挡  
                    if (curr_range < next_range) {
                        disable_point[j + 5] = 1;
                        disable_point[j + 4] = 1;
                        disable_point[j + 3] = 1;
                        disable_point[j + 2] = 1;
                        disable_point[j + 1] = 1;
                        j = j + 4;  
                    } else {
                        disable_point[j] = 1;
                        disable_point[j - 1] = 1;
                        disable_point[j - 2] = 1;
                        disable_point[j - 3] = 1;
                        disable_point[j - 4] = 1;
                        disable_point[j - 5] = 1;
                    } 
                }
                // 检测平行
                // 前后点均连续 且range的差异是递增的
            }
            LOG(INFO) << "un-continue n:"<<n; 
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 计算每一层scan的起始index, 并不关心到底是哪一层 
         * @details: 输入点云需要保证有ring信息，要求 pc_in.points 中点云的排列是按照scan有序分布的
         * @param[in] pc_in 输入点云
         * @param[out] scan_index 保存各个scan层点index的容器 , 并且是按顺序排列的
         */            
        void splitScan(InputPointCloud const& pc_in, std::vector<uint32_t> &scan_index) {
            scan_index.clear();
            scan_index.reserve(128);
            std::vector<bool> scan_flag; // 记录scan是否已经访问
            scan_flag.resize(128, 0);
            uint16_t  last_id = -1;
            for (uint32_t i = 0; i < (uint32_t)pc_in.points.size(); i++) {
                uint16_t const& scan_id = pc_in.points[i].ring; 
                // 说明层变化了
                if (scan_id != last_id) {
                    // 层错乱了
                    if (scan_flag[scan_id]) {
                        throw "点云ring混乱";
                    }
                    scan_index.push_back(i);
                    scan_flag[scan_id] = 1; 
                    last_id = scan_id; 
                }
            }
        }

    private:
        Option option_;   
        bool check_bad_points_; 
        float rotation_angle_v_; // 旋转角速度 
        float horizon_angle_resolution_rad_;              
        // VoxelGridFilter<_OutputFeatureT> down_sampling_edge_; 
        // VoxelGridFilter<_OutputFeatureT> down_sampling_surf_;  
}; // class LOAMFeatureExtractor
} // namespace 
}