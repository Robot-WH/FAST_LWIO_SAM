/*
 * @Copyright(C): 
 * @Author: lwh
 * @Date: 2022-04-09 17:00:25
 * @Description:  基于聚类的点云分割 
 * @Others: 通用的方法 
 */

#pragma once
#include "Common/pcl_type.h"
#include "../GroundDetect/AngleBasedGroundDetect.hpp"
#include "../Cluster/DbscanCluster.hpp"
#include <opencv/cv.hpp>

namespace Slam3D {
namespace Algorithm {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: 点云分割 
 * @details: 地面分割 + 欧式聚类  
 */    
template<typename _PointT>
class PointCloudSegmentation {
    public:
        #define DEBUG 0 
        // 配置参数 
        struct Option {
            bool debug_ = false; 
            bool has_ring_ = false; 
            uint16_t line_num_ = 16;    // 线数    默认16 
            uint16_t ground_line_num_ = 8;    // 地面线数
            float horizon_angle_resolution_ = 0.2;      // 0.2 10hz  0.386 19.3hz 

            float segment_angle_thresh_ = 0.17;   // 10度  10 * 3.14 / 180 = 0.17
            float clustered_num_thresh_ = 30;    // 聚类成立的数量阈数
            float min_valid_num_ = 10;    // 聚类成立的数量阈数
            float min_vertical_line_num_ = 5; 
        };

        struct ClusterInfo {
            Eigen::Vector3d direction_max_value_{0, 0, 0};
            Eigen::Vector3d centre_{0, 0, 0};
            Eigen::Vector3d direction_{0, 0, 0};
        };

        enum ClusterType{
            STABLE_POINT = 1,   // 若聚类后  聚类体的size足够大  那么认为是稳定的
            UNSTABLE_POINT,  // 聚类后，size 不够大，则认为是潜在的动态点 
            GROUND_POINT
        };

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // 构造一：外部进行参数配置 然后传入参数进行初始化
        PointCloudSegmentation(Option const& option) : option_(option) {
            Init();
        }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // 构造二：传入参数文件地址  直接读参数文件初始化参数 
        PointCloudSegmentation(std::string config_path) {}

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void Init() {
            ground_pointcloud_.reset(new pcl::PointCloud<_PointT>());
            nonground_pointcloud_.reset(new pcl::PointCloud<_PointT>());
            cluster_info_container_.reserve(2000); 
            // 地面提取
            typename AngleBasedGroundDetect<_PointT>::Option ground_detect_option;
            ground_detect_option.line_num_ = option_.line_num_;
            ground_detect_option.ground_line_num_ = option_.ground_line_num_;
            ground_detect_option.horizon_angle_resolution_ = option_.horizon_angle_resolution_;
            ground_detect_.reset(new AngleBasedGroundDetect<_PointT>(ground_detect_option));  
            // dbscan   
            typename DBSCANCluster<_PointT>::Option cluster_option;
            cluster_option.search_range_coeff_ = 0.06;
            cluster_option.max_search_range = 1.2;
            cluster_.reset(new DBSCANCluster<_PointT>(cluster_option));  

            resetData(); 
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void resetData() {
            stable_pointcloud_.clear(); 
            unstable_pointcloud_.clear(); 
            ground_pointcloud_->clear(); 
            nonground_pointcloud_->clear(); 
            outlier_pointcloud_.clear(); 
            cluster_info_container_.clear(); 
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        virtual void Process(pcl::PointCloud<_PointT> const& cloud_in) {
            resetData(); 
            ground_detect_->GroundDetect(cloud_in);  // 地面检测 
            cloudSegmentation(cloud_in);  // 聚类分割 
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        pcl::PointCloud<_PointT> const& GetNonGroundPoints() {
            return *nonground_pointcloud_;
        }   

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        pcl::PointCloud<_PointT> const& GetGroundPoints() {
            return *ground_pointcloud_;
        }   

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        pcl::PointCloud<_PointT> const& GetStablePoints() {
            return stable_pointcloud_;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        pcl::PointCloud<_PointT> const& GetUnStablePoints() {
            return unstable_pointcloud_;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        pcl::PointCloud<_PointT> const& GetOutlierPoints() {
            return outlier_pointcloud_;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        std::vector<ClusterInfo> const& GetClusterInfo() {
            return cluster_info_container_;
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void SetRingFlag(bool flag) {
            option_.has_ring_ = flag;
        }

    protected:

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 点云分割  
         */        
        void cloudSegmentation(pcl::PointCloud<_PointT> const& cloud_in) {
            // 分离出地面点与非地面点  
            std::vector<uint32_t> const& ground_index = ground_detect_->GetIndices(); 
            std::vector<uint32_t> label_vec; 
            label_vec.resize(cloud_in.size(), 0); 
            for (auto const& i : ground_index) {
                ground_pointcloud_->push_back(cloud_in.points[i]);
                label_vec[i] = 1;
            }
            for (uint32_t i = 0; i < cloud_in.size(); i++) {
                if (!label_vec[i]) {
                    nonground_pointcloud_->push_back(cloud_in.points[i]);
                }
            }
            // 提取聚类   合格的聚类index保存在 cluster_indices 中 
            std::vector<std::vector<uint32_t>> cluster_indices;
            std::vector<std::vector<uint32_t>> outlier_indices;   
            cluster_indices.reserve(9999);
            outlier_indices.reserve(9999);
            TicToc tt; 
            cluster_->Extract(nonground_pointcloud_, cluster_indices, outlier_indices);
            for (auto const& cluster : cluster_indices) {
                ClusterInfo info;
                // 获取聚类的物理尺度   判断是否是背景点还是前景点 
                pca(*nonground_pointcloud_, cluster,  info);
                if (info.direction_max_value_[2] < 5) {
                    cluster_info_container_.push_back(std::move(info));
                    // 提取点云
                    for (auto const& i : cluster) {
                        unstable_pointcloud_.push_back(nonground_pointcloud_->points[i]); 
                    }
                } else {
                    // 提取点云
                    for (auto const& i : cluster) {
                        stable_pointcloud_.push_back(nonground_pointcloud_->points[i]); 
                    }
                }
            }  
            for (auto const& outlier : outlier_indices) {
                // 提取点云
                for (auto const& i : outlier) {
                    outlier_pointcloud_.push_back(nonground_pointcloud_->points[i]); 
                }
            }
        }  

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void pca(pcl::PointCloud<_PointT> const& cloud_in, 
                std::vector<uint32_t> const& index,  ClusterInfo &info) {
            std::vector<Eigen::Vector3d> points;
            std::vector<Eigen::Vector3d> decentration_points;   // 去中心化
            info.centre_ = {0, 0, 0};
            uint16_t size = index.size();  
            points.reserve(size);
            decentration_points.reserve(size);

            for (int32_t const& i : index) {
                Eigen::Vector3d point(cloud_in.points[i].x, cloud_in.points[i].y, cloud_in.points[i].z);
                info.centre_ += point;
                points.push_back(std::move(point));
            }
            info.centre_ /= size;
            // 计算协方差矩阵
            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
            for (int32_t i = 0; i < size; i++) {
                Eigen::Vector3d tmpZeroMean = points[i] - info.centre_;
                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                decentration_points.push_back(std::move(tmpZeroMean));
            }
            covMat /= size;  
            // 特征分解 
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
            info.direction_max_value_ = {0, 0, 0};
            auto eigen_vector = saes.eigenvectors(); 
            // 将各个点向主方向投影
            for (uint8_t i = 2; i < 3; i++) {
                Eigen::Vector3d direction = eigen_vector.col(i);
                for (int32_t j = 0; j < size; j++) {
                    double value = decentration_points[j].transpose() * direction; 
                    if (value > info.direction_max_value_[i]) {
                        info.direction_max_value_[i] = value;  
                    }   
                }
            }
        }

    private:
        Option option_;  
        typename AngleBasedGroundDetect<_PointT>::Ptr ground_detect_;    // 地面检测
        typename pcl::PointCloud<_PointT>::Ptr ground_pointcloud_;
        typename pcl::PointCloud<_PointT>::Ptr nonground_pointcloud_;
        pcl::PointCloud<_PointT> stable_pointcloud_;
        pcl::PointCloud<_PointT> unstable_pointcloud_;
        pcl::PointCloud<_PointT> outlier_pointcloud_; 
        std::vector<ClusterInfo> cluster_info_container_;
        typename DBSCANCluster<_PointT>::Ptr cluster_; 
}; // class 
} // namespace 
}

