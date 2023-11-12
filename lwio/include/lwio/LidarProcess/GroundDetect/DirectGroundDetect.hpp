/*
 * @Copyright(C): 
 * @Author: lwh
 * @Description: 直接地面检测      1、根据高度先验提取出候选地面点   2、ransac地面拟合
 * @Others: 
 */
#pragma once
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include "SlamLib/tic_toc.hpp"

namespace lwio {

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: 基于角度的地面检测方法
 * @details: ref.legoloam，每一帧的点云需要是规则有序的(旋转式激光雷达)
 */    
template<typename _PointT>
class DirectGroundDetect {
public:
    using Ptr = std::unique_ptr<DirectGroundDetect<_PointT>>;

    struct Option {
        uint16_t ground_points_num_thresh_ = 500;
        float max_height_ = 0; // 地面的最大高度   
        float fitting_threshold_ = 0.3;  
    };
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    DirectGroundDetect(Option const& option) : option_(option) {
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief: 地面提取
     * @param cloud_in 输入的点云
     * @param ground_indices 地面点在原始点云中的序号 
     * @param index_image x,y处的值为 点云图像x,y坐标处的点在输入点云中的index
     * @return {*}
     */        
    void GroundDetect(pcl::PointCloud<_PointT>& cloud_in, 
            std::vector<uint32_t>& ground_indices,
            pcl::PointCloud<_PointT>& ground_cloud) {
        ground_indices.clear(); 
        ground_cloud.clear();  
        std::vector<uint32_t> candidata_indices;   // 候选地面点的序号 
        candidata_indices.reserve(cloud_in.size());
        ground_indices.reserve(cloud_in.size());
        ground_cloud.reserve(cloud_in.size());
        
        SlamLib::PCLPtr<_PointT> candidata_pointcloud(new pcl::PointCloud<_PointT>());
        candidata_pointcloud->reserve(cloud_in.size());
    
        // step1、分割出候选地面点
        for (uint32_t j = 0; j < cloud_in.size(); ++j) {
            if (cloud_in[j].z < option_.max_height_) {
                candidata_indices.push_back(j);
                candidata_pointcloud->push_back(cloud_in[j]); 
            }
        }
        // std::cout << "candidata_pointcloud size: " << candidata_pointcloud->size() << std::endl;
        // step2、基于RANSAC优化地面并且拟合出地面参数 
        typename pcl::SampleConsensusModelPlane<_PointT>::Ptr model_p(
            new pcl::SampleConsensusModelPlane<_PointT>(candidata_pointcloud));

        pcl::RandomSampleConsensus<_PointT> ransac(model_p);
        ransac.setDistanceThreshold(option_.fitting_threshold_);     // 与该平面距离小于该阈值的点为内点 
        ransac.computeModel();
        // std::cout << "computeModel" << std::endl;
        // 获取内点  
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        ransac.getInliers(inliers->indices);
        // 提取出地面信息
        for(int& i : inliers->indices) {
            ground_indices.push_back(candidata_indices[i]); 
            ground_cloud.push_back(cloud_in[candidata_indices[i]]); 
            cloud_in[candidata_indices[i]].type = 3;   // 标记为地面点  
        }
        // 对地面进行评估
        // 地面点数量
        if(ground_cloud.size() < option_.ground_points_num_thresh_) {
            ground_coeffs_ = {0., 0., 0., 0.};
            return;    // 地面点数量不合格  
        }
        // // 平面质量验证
        // Eigen::VectorXf coeffs;
        // ransac.getModelCoefficients(coeffs);
        // int bad_num = 0, good_num = 0; 
        // float sum_dis = 0; 
        // int bad_thresh = inliers->indices.size() * 0.3; // 距离过大的点的数量阈值    超过这个值 地面就不理想了
        // int good_thresh = inliers->indices.size() * 0.7;  // 距离合适的点的数量阈值  
        // // 遍历全部内点 
        // for (int& i:inliers->indices) {
        //     _PointT const& point = candidata_pointcloud_->points[i];
        //     float dis = fabs(point.x * coeffs[0] + point.y * coeffs[1] + point.z * coeffs[2] + coeffs[3]);     // 最好->0  
        //     // 不好的地面点
        //     if(dis > 0.1) {
        //         bad_num++;
        //         if (bad_num > bad_thresh) {
        //             ground_coeffs_ = {0., 0., 0., 0.};
        //             return;    // 地面质量不合格  
        //         }
        //     } else {
        //         sum_dis += dis;  
        //         good_num++; 
        //         if (good_num > good_thresh) {
        //             break; 
        //         }
        //     }
        // }
        // quality_score_ = sum_dis / good_num; 
        // std::cout<<"floor detect: "<<coeffs.transpose()<<std::endl;
        // std::cout<<"quality_score_: "<<quality_score_<<std::endl;
        // if(!floor_init)
        // {
        //     floor_init = true;
        //     Eigen::Vector3d z = {0,0,-1};
        //     Eigen::Vector3d e = {coeffs[0],coeffs[1],coeffs[2]};
        //     // 求旋转法向量   
        //     Eigen::Vector3d n = z.cross(e);
        //     double theta = atan2( n.norm(), z.dot(e));       // 旋转角  
        //     n.normalize();       // 单位化
        //     tilt_matrix.topLeftCorner(3, 3) = Eigen::AngleAxisd(theta, n).toRotationMatrix().inverse();    // 计算校正矩阵  
        //     std::cout<<"floor init ok !  tilt theta: "<<theta<<" n: "<<n<<std::endl;
        //     std::cout<<"check !  correct: "<<tilt_matrix*coeffs.cast<double>()<<std::endl;
        // }
    }


private: 
    Option option_; 
    Eigen::Vector4f ground_coeffs_;   // 地面参数 
    float quality_score_;  // 地面质量得分
}; // class
}
