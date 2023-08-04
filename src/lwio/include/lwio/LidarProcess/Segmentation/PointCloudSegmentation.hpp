/*
 * @Copyright(C): 
 * @Author: lwh
 * @Date: 2022-04-09 17:00:25
 * @Description:  基于聚类的点云分割 
 * @Others: 通用的方法 
 */
#pragma once
#include <yaml-cpp/yaml.h>
#include <opencv/cv.hpp>
#include "../GroundDetect/DirectGroundDetect.hpp"
#include "SlamLib/PointCloud/Cluster/DbscanCluster.h"


namespace lwio {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: 点云分割 
 * @details: 地面分割 + 聚类  
 */    
template<typename _PointT>
class PointCloudSegmentation {
public:
    #define DEBUG 0 
    // 配置参数 
    struct Option {
        bool debug_ = false; 
        bool has_ring_ = false; 
    };

    struct ClusterInfo {
        Eigen::Vector3d direction_max_value_{0, 0, 0};
        Eigen::Vector3d centre_{0, 0, 0};
        Eigen::Vector3d direction_{0, 0, 0};
    };

    enum ClusterType {
        STABLE_POINT = 1,   // 若聚类后  聚类体的size足够大  那么认为是稳定的
        UNSTABLE_POINT,  // 聚类后，size 不够大，则认为是潜在的动态点 
        GROUND_POINT
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 构造二：传入参数文件地址  直接读参数文件初始化参数 
    PointCloudSegmentation(std::string config_path) {
        YAML::Node yaml = YAML::LoadFile(config_path);
        ground_pointcloud_.reset(new pcl::PointCloud<_PointT>());
        nonground_pointcloud_.reset(new pcl::PointCloud<_PointT>());
        cluster_info_container_.reserve(2000); 
        // 地面提取
        typename DirectGroundDetect<_PointT>::Option ground_detect_option;
        ground_detect_.reset(new DirectGroundDetect<_PointT>(ground_detect_option));  
        // dbscan   
        typename SlamLib::pointcloud::DBSCANCluster<_PointT>::Option cluster_option;
        cluster_option.search_range_coeff_ = 0.6;
        cluster_option.max_search_range = 1.5;
        // 最小的聚类搜索范围为 sqrt(3) * 降采样网格分辨率
        float downsample_resolution = 
                    yaml["preprocess"]["downsampling"]["VoxelGrid"]["resolution"].as<float>();
        cluster_option.min_search_range = std::sqrt(3) * downsample_resolution;  
        cluster_.reset(new SlamLib::pointcloud::DBSCANCluster<_PointT>(cluster_option));  

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
    virtual void Process(SlamLib::PCLPtr<_PointT>& cloud_in,
                                                SlamLib::PCLPtr<_PointT>& unstable_points,
                                                SlamLib::PCLPtr<_PointT>& stable_points,
                                                SlamLib::PCLPtr<_PointT>& ground_points) {
        resetData(); 
        std::vector<uint32_t> ground_index; 
        ground_detect_->GroundDetect(*cloud_in, ground_index, *ground_points);  // 地面检测 
        // cloudSegmentation(*cloud_in, ground_index, unstable_points, stable_points);  // 聚类分割
        // std::cout << "stable_points size: " << stable_points->size() << std::endl; 
        // std::cout << "unstable_points size: " << unstable_points->size() << std::endl; 
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
    void cloudSegmentation(const pcl::PointCloud<_PointT>& cloud_in, 
            const std::vector<uint32_t>&ground_index, 
            SlamLib::PCLPtr<_PointT>& unstable_points,
            SlamLib::PCLPtr<_PointT>& stable_points) {
        // 分离出地面点与非地面点  
        std::vector<uint32_t> label_vec; 
        label_vec.resize(cloud_in.size(), 0); 
        for (auto const& i : ground_index) {
            label_vec[i] = 1;
        }
        for (uint32_t i = 0; i < cloud_in.size(); i++) {
            if (!label_vec[i]) {
                nonground_pointcloud_->push_back(cloud_in.points[i]);
            }
        }
        stable_points->reserve(nonground_pointcloud_->size());
        unstable_points->reserve(nonground_pointcloud_->size()); 
        // 提取聚类   合格的聚类index保存在 cluster_indices 中 
        std::vector<std::vector<uint32_t>> cluster_indices;
        std::vector<std::vector<uint32_t>> outlier_indices;   
        cluster_indices.reserve(9999);
        outlier_indices.reserve(9999);
        SlamLib::time::TicToc tt; 
        cluster_->Extract(nonground_pointcloud_, cluster_indices, outlier_indices);
        
        for (auto const& cluster : cluster_indices) {
            ClusterInfo info;
            // 获取聚类的物理尺度   判断是否是背景点还是前景点 
            pca(*nonground_pointcloud_, cluster,  info);
            if (info.direction_max_value_[2] < 5) {
                cluster_info_container_.push_back(std::move(info));
                // 提取点云
                for (auto const& i : cluster) {
                    unstable_points->push_back(nonground_pointcloud_->points[i]); 
                }
            } else {
                // 提取点云
                for (auto const& i : cluster) {
                    stable_points->push_back(nonground_pointcloud_->points[i]); 
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
    typename SlamLib::pointcloud::DBSCANCluster<_PointT>::Ptr cluster_; 
    typename DirectGroundDetect<_PointT>::Ptr ground_detect_;    // 地面检测
    typename pcl::PointCloud<_PointT>::Ptr ground_pointcloud_;
    typename pcl::PointCloud<_PointT>::Ptr nonground_pointcloud_;
    pcl::PointCloud<_PointT> stable_pointcloud_;
    pcl::PointCloud<_PointT> unstable_pointcloud_;
    pcl::PointCloud<_PointT> outlier_pointcloud_; 
    std::vector<ClusterInfo> cluster_info_container_;
}; // class 
} // namespace 

