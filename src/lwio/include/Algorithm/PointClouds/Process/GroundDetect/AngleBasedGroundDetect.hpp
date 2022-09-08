/*
 * @Copyright(C): 
 * @Author: lwh
 * @Description: 基于距离图像的地面检测
 * @Others: 
 */
#pragma once
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include "../LidarModel/RotatingLidarModel.hpp"
#include "tic_toc.h"

namespace Slam3D {
namespace Algorithm {

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: 基于角度的地面检测方法
 * @details: ref.legoloam，每一帧的点云需要是规则有序的(旋转式激光雷达)
 */    
template<typename _PointT>
class AngleBasedGroundDetect {
    public:
        using Ptr = std::unique_ptr<AngleBasedGroundDetect<_PointT>>;

        struct Option {
            uint16_t line_num_ = 16;    // 线数    默认16 
            uint16_t ground_line_num_ = 8;    // 地面线数
            float horizon_angle_resolution_ = 0.2;      // 0.2 10hz  0.386 19.3hz 
            uint16_t ground_points_num_thresh_ = 1000;
            float ground_angle_thresh_ = 10; // 地面观测角度 
        };
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        AngleBasedGroundDetect(Option const& option) : option_(option) {
            init();  
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // AngleBasedGroundDetect(Option const& option, RotatingLidarModel<_PointT> const& model) 
        // : option_(option), lidar_model_(model) {
        //     init(); 
        // }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 地面提取
         * @param cloud_in 输入的点云
         * @return {*}
         */        
        void GroundDetect(pcl::PointCloud<_PointT> const& cloud_in) {
            /**
             * @todo : 玄学 ！！！！ 将 index_img 改为 局部变量 cv::Mat index_img 将出线周期性耗时增加的问题 
             */
            index_img = cv::Mat(option_.ground_line_num_, SCAN_NUM_, CV_32S, cv::Scalar::all(-1));
            // 先计算出点云模型的index_image
            calcIndexImage(cloud_in, index_img);
            GroundDetect(cloud_in, index_img);
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 地面提取
         * @param cloud_in 输入的点云
         * @param index_image x,y处的值为 点云图像x,y坐标处的点在输入点云中的index
         * @return {*}
         */        
        void GroundDetect(pcl::PointCloud<_PointT> const& cloud_in, cv::Mat const& index_image) {
            resetData(); 
            int32_t lowerInd, upperInd;
            float diffX, diffY, diffZ, angle;
            ground_indices_.clear(); 
            std::vector<uint32_t> candidata_indices_;   // 候选地面点的序号 
            std::vector<std::pair<uint16_t, uint16_t>> candidata_pos_;   // 候选地面点的位置 
            candidata_indices_.reserve(option_.ground_line_num_ * SCAN_NUM_);
            candidata_pos_.reserve(option_.ground_line_num_ * SCAN_NUM_);
            typename pcl::PointCloud<_PointT>::Ptr candidata_pointcloud_(new pcl::PointCloud<_PointT>());
            // step1、分割出候选地面点
            for (uint16_t j = 0; j < SCAN_NUM_; ++j) {
                for (uint16_t i = 0; i < option_.ground_line_num_ - 1; ++i) {
                    lowerInd = index_image.at<int32_t>(i, j);
                    upperInd = index_image.at<int32_t>(i + 1, j);
                    // 无点
                    if (lowerInd == -1 || upperInd == -1) {
                        ground_image_.at<int8_t>(i, j) = -1;
                        continue;
                    }
                    diffX = cloud_in.points[upperInd].x - cloud_in.points[lowerInd].x;
                    diffY = cloud_in.points[upperInd].y - cloud_in.points[lowerInd].y;
                    diffZ = cloud_in.points[upperInd].z - cloud_in.points[lowerInd].z;
                    angle = atan2(diffZ, sqrt(diffX * diffX + diffY * diffY)) * 180 / M_PI;
                    // 默认激光经过 校正  点云与地面是平行的
                    // 地面最多10度的起伏
                    if (abs(angle) <= option_.ground_angle_thresh_) {
                        candidata_pos_.emplace_back(i, j);
                        candidata_pos_.emplace_back(i + 1, j);
                        candidata_indices_.push_back(lowerInd);
                        candidata_indices_.push_back(upperInd);
                        candidata_pointcloud_->push_back(cloud_in.points[lowerInd]); 
                        candidata_pointcloud_->push_back(cloud_in.points[upperInd]); 
                    } 
                }
            }
            // step2、基于RANSAC优化地面并且拟合出地面参数 
            typename pcl::SampleConsensusModelPlane<_PointT>::Ptr model_p(
                new pcl::SampleConsensusModelPlane<_PointT>(candidata_pointcloud_));
            pcl::RandomSampleConsensus<_PointT> ransac(model_p);
            ransac.setDistanceThreshold(0.2);     // 与该平面距离小于该阈值的点为内点 
            ransac.computeModel();
            // 获取内点  
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            ransac.getInliers(inliers->indices);
            // 提取出地面信息
            for(int& i : inliers->indices) {
                ground_indices_.push_back(candidata_indices_[i]); 
                std::pair<uint16_t, uint16_t> const& pos = candidata_pos_[i];
                ground_image_.at<int8_t>(pos.first, pos.second) = 1;
            }
            // 对地面进行评估
            // 地面点数量
            if(inliers->indices.size() < option_.ground_points_num_thresh_) {
                ground_coeffs_ = {0., 0., 0., 0.};
                return;    // 地面点数量不合格  
            }
            // 平面质量验证
            Eigen::VectorXf coeffs;
            ransac.getModelCoefficients(coeffs);
            int bad_num = 0, good_num = 0; 
            float sum_dis = 0; 
            int bad_thresh = inliers->indices.size() * 0.3; // 距离过大的点的数量阈值    超过这个值 地面就不理想了
            int good_thresh = inliers->indices.size() * 0.7;  // 距离合适的点的数量阈值  
            // 遍历全部内点 
            for (int& i:inliers->indices) {
                _PointT const& point = candidata_pointcloud_->points[i];
                float dis = fabs(point.x * coeffs[0] + point.y * coeffs[1] + point.z * coeffs[2] + coeffs[3]);     // 最好->0  
                // 不好的地面点
                if(dis > 0.1) {
                    bad_num++;
                    if (bad_num > bad_thresh) {
                        ground_coeffs_ = {0., 0., 0., 0.};
                        return;    // 地面质量不合格  
                    }
                } else {
                    sum_dis += dis;  
                    good_num++; 
                    if (good_num > good_thresh) {
                        break; 
                    }
                }
            }
            quality_score_ = sum_dis / good_num; 
            // std::cout<<"floor detect: "<<coeffs.transpose()<<std::endl;
            // std::cout<<"quality_score_: "<<quality_score_<<std::endl;
            // std::cout<<"num: "<<num<<std::endl;
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

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        cv::Mat const& GetGroundImage() const {
            return ground_image_; 
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        std::vector<uint32_t> const& GetIndices() const {
            return ground_indices_;  
        }

    private:
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void init() {
            SCAN_NUM_ = std::ceil(360 / option_.horizon_angle_resolution_);  
            ground_indices_.reserve(option_.ground_line_num_ * SCAN_NUM_);
            // 构造旋转式激光雷达模型
            typename RotatingLidarModel<_PointT>::Option model_option;
            model_option.line_num_ = option_.line_num_;
            model_option.horizon_angle_resolution_ = option_.horizon_angle_resolution_;          
            lidar_model_.reset(new RotatingLidarModel<_PointT>(model_option));
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void resetData() {
            ground_image_ = cv::Mat(option_.ground_line_num_, SCAN_NUM_, CV_8S, cv::Scalar::all(0));
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 计算旋转雷达序号图像  
         */        
        void calcIndexImage(pcl::PointCloud<_PointT> const& laserCloudIn, cv::Mat &index_img) {
            int16_t rowIdn;
            int32_t columnIdn;
            _PointT thisPoint;
            int32_t cloudSize = laserCloudIn.points.size();

            for (int32_t i = 0; i < cloudSize; ++i) {
                thisPoint.x = laserCloudIn.points[i].x;
                thisPoint.y = laserCloudIn.points[i].y;
                thisPoint.z = laserCloudIn.points[i].z;
                thisPoint.range = laserCloudIn.points[i].range;
                // 使用雷达模型获取激光点的线数和列数
                rowIdn = lidar_model_->GetRingID(thisPoint); 
                if (rowIdn < 0 || rowIdn >= option_.ground_line_num_) continue; 
                columnIdn = lidar_model_->GetColumn(thisPoint).second; 
                if (columnIdn < 0) continue;  
                index_img.at<int32_t>(rowIdn, columnIdn) = i;  
            }
            return;
        }

    private: 
        Option option_; 
        cv::Mat ground_image_;
        cv::Mat index_img;
        std::vector<uint32_t> ground_indices_;   // 地面点的序号
        std::unique_ptr<RotatingLidarModel<_PointT>> lidar_model_;   
        Eigen::Vector4f ground_coeffs_;   // 地面参数 
        float quality_score_;  // 地面质量得分
        uint16_t SCAN_NUM_;
}; // class
} 
}
