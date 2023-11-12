#pragma once 
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>
#include <flann/flann.hpp>
#include <opencv2/opencv.hpp>
#include "LidarTrackerBase.hpp"
#include "SlamLib/Map/LocalMap.hpp"
#include "SlamLib/Map/TimedSlidingLocalMap.hpp"
#include "SlamLib/Map/iVoxTimedSlidingLocalMap.hpp"
#include "SlamLib/PointCloud/Registration/ceres_edgeSurfFeatureRegistration.h"
#include "SlamLib/PointCloud/Registration/edgeSurfFeatureRegistration.h"
#include "SlamLib/Common/pointcloud.h"
#include "SlamLib/tic_toc.hpp"
#include "SlamLib/PointCloud/Cluster/EuclideanCluster.h"
#include "lwio/Common/color.hpp"

namespace lwio {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: tracker LoclMap匹配模式框架
 * @details:  适应多种Local Map构造方式 ， 以及多种匹配方式  a. 特征ICP  b. NDT/GICP
 * @param _PointType 使用的特征点的类型  
 */    
template<typename _PointType>
class LidarTracker : public LidarTrackerBase<_PointType> {
private:
    using LocalMapPtr =  typename SlamLib::map::PointCloudLocalMapBase<_PointType>::Ptr;
    using RegistrationResult = typename SlamLib::pointcloud::RegistrationBase<_PointType>::RegistrationResult; 
    using PointVector = typename SlamLib::pointcloud::RegistrationBase<_PointType>::PointVector;  
    enum class LocalMapUpdataType {
        NO_UPDATA = 0,
        MOTION_UPDATA,
        TIME_UPDATA 
    };
public:
    struct Option {
        float THRESHOLD_TRANS_;
        float THRESHOLD_ROT_;  
        float TIME_INTERVAL_;  
    };
    struct rangeImageOption {
        uint16_t h_angle_res_ = 1;    // 水平角度分辨率  单位：度  
        uint16_t v_angle_res_ = 10;  // 垂直角分辨率  
        float h_angle_res_rad_ = h_angle_res_ * M_PI / 180;    // 水平角度分辨率  单位：度  
        float v_angle_res_rad_ = v_angle_res_ * M_PI / 180;  // 垂直角分辨率  
        uint16_t h_size_ = 360 / h_angle_res_;     
        uint16_t v_size_ = 180 / v_angle_res_;   // 垂直视角  +-90 
    }; 

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief: 传入参数文件地址，直接读取参数并初始化
     */        
    LidarTracker(const std::string& param_path)  : init_(false), registration_ptr_(nullptr) {
        try {
            YAML::Node yaml = YAML::LoadFile(param_path);
            LOG(INFO) << "initialize LidarTracker..."; 

            // 先设置匹配算法 
            std::string registration_method = yaml["tracker"]["registration"]["method"].as<std::string>();
            LOG(INFO) << "registration_method:"<<registration_method; 

            if (registration_method == "point_plane_icp") {
                std::string solve = yaml["tracker"]["registration"]["point_plane_icp"]["solve"].as<std::string>();
                LOG(INFO) << "solve:"<<solve; 
                std::string point_label = yaml["tracker"]["registration"]["point_plane_icp"]["point_label"].as<std::string>();
                LOG(INFO) << "point_label:"<<point_label; 
                if (solve == "ceres") {
                    registration_ptr_.reset(new SlamLib::pointcloud::CeresEdgeSurfFeatureRegistration<_PointType>("", point_label)); 
                } else if (solve == "LM" || solve == "GN") {
                    typename SlamLib::pointcloud::EdgeSurfFeatureRegistration<_PointType>::Option option;
                    option.surf_label_ = point_label;
                    option.max_iterater_count_ = 
                        yaml["tracker"]["registration"]["point_plane_icp"]["max_iterater_count"].as<int>();
                    option.norm_iterater_count_ = 
                        yaml["tracker"]["registration"]["point_plane_icp"]["norm_iterater_count"].as<int>();

                    if (solve == "GN") {
                        option.method_ = SlamLib::pointcloud::EdgeSurfFeatureRegistration<_PointType>::OptimizeMethod::GN;
                        option.gn_option_.max_iterater_count_ = 
                            yaml["tracker"]["registration"]["point_plane_icp"]["GN"]["max_iterater_count"].as<int>();
                    } else {
                        option.method_ = SlamLib::pointcloud::EdgeSurfFeatureRegistration<_PointType>::OptimizeMethod::LM;
                        option.lm_option_.max_iterater_count_ = 
                            yaml["tracker"]["registration"]["point_plane_icp"]["LM"]["max_iterater_count"].as<int>();
                    }

                    registration_ptr_.reset(new SlamLib::pointcloud::EdgeSurfFeatureRegistration<_PointType>(option)); 
                }
            } else if (registration_method == "ndt") {
            } else if (registration_method == "gicp") {
            }
            // LOG(INFO) << common::GREEN << "registration_method init done"
            //     << common::RESET; 
            // 设置Local map 
            std::string localmap_method = yaml["tracker"]["localmap"]["method"].as<std::string>();
            LOG(INFO) << "localmap_method:"<<localmap_method; 
            if (localmap_method == "time_sliding") {
                typename SlamLib::map::TimedSlidingLocalMap<_PointType>::Option option;  
                option.window_size_ = yaml["tracker"]["localmap"]["time_sliding"]["window_size"].as<int>();
                option.use_kdtree_search_ = yaml["tracker"]["localmap"]["time_sliding"]["kdtree_enable"].as<bool>();
                local_map_.reset(new SlamLib::map::TimedSlidingLocalMap<_PointType>(option));
            } else if (localmap_method == "area_sliding") {
            } else if (localmap_method == "ivox") {
            } else if (localmap_method == "timedIvox") {
                typename SlamLib::map::IvoxTimedSlidingLocalMap<_PointType>::Option option; 
                option.window_size_ = yaml["tracker"]["localmap"]["timedIvox"]["window_size"].as<float>();
                LOG(INFO) << "timedIvox map window_size:"<<option.window_size_; 
                option.ivox_option_.resolution_ = yaml["tracker"]["localmap"]["timedIvox"]["resolution"].as<float>();
                LOG(INFO) << "ivox resolution:"<<option.ivox_option_.resolution_; 
                option.ivox_option_.inv_resolution_ = 1 / option.ivox_option_.resolution_;  
                option.ivox_option_.capacity_ = yaml["tracker"]["localmap"]["timedIvox"]["capacity"].as<int>();
                LOG(INFO) << "ivox capacity:"<<option.ivox_option_.capacity_; 
                int nearby_type = yaml["tracker"]["localmap"]["timedIvox"]["nearby_type"].as<int>();
                if (nearby_type == 18) {
                    LOG(INFO) << "ivox nearby_type: NEARBY18"; 
                    option.ivox_option_.nearby_type_ = SlamLib::map::IvoxTimedSlidingLocalMap<_PointType>::IvoxNearbyType::NEARBY18;
                }
                local_map_.reset(new SlamLib::map::IvoxTimedSlidingLocalMap<_PointType>(option));
            }
            // 设置关键帧提取参数
            option_.THRESHOLD_TRANS_ = yaml["tracker"]["keyframe_update"]["translation"].as<float>();
            LOG(INFO) << "local map update translation:"<<option_.THRESHOLD_TRANS_;
            option_.THRESHOLD_ROT_ = yaml["tracker"]["keyframe_update"]["rotation"].as<float>();
            LOG(INFO) << "local map update rotation:"<<option_.THRESHOLD_ROT_;
            option_.TIME_INTERVAL_ = yaml["tracker"]["keyframe_update"]["time"].as<float>();
            LOG(INFO) << "local map update time:"<<option_.TIME_INTERVAL_;

            updata_type_ = LocalMapUpdataType::NO_UPDATA; 

            dynamic_removal_ = yaml["tracker"]["dynamic_removal"]["status"].as<bool>();
            if (dynamic_removal_) {
                std::cout << "开启动态检测及滤除" << std::endl;
            } else {
                std::cout << "关闭动态检测及滤除" << std::endl;
            }
            std::string lidar_type = yaml["tracker"]["dynamic_removal"]["lidar_type"].as<std::string>();
            if (lidar_type == "unrotary") {
                view_limited_ = true; 
            }
        } catch (const YAML::Exception& e) {
            LOG(ERROR) << "YAML Exception: " << e.what();
        }

        dynamic_cloud_.reset(new pcl::PointCloud<_PointType>());
        false_dynamic_cloud_.reset(new pcl::PointCloud<_PointType>());
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief: 激光tracker 求解 
     * @details:  求解出当前激光与上一帧激光数据的增量  
     * @param[in] data 特征点云数据
     * @param[in] timestamp 时间戳  
     * @param[out] deltaT 输出的增量
     */            
    virtual void Solve(const SlamLib::CloudContainer<_PointType>& data, 
            Eigen::Isometry3d& deltaT) override {
        // local map 初始化
        if (init_ == false) {   
            updateLocalMap(data.feature_data_, Eigen::Isometry3d::Identity(), 
                LocalMapUpdataType::TIME_UPDATA);
            curr_pose_ = Eigen::Isometry3d::Identity();  
            prev_pose_ = Eigen::Isometry3d::Identity();  
            motion_increment_ = Eigen::Isometry3d::Identity();  
            last_keyframe_pose_ = Eigen::Isometry3d::Identity();  
            deltaT.setIdentity();  
            last_keyframe_time_ = data.timestamp_start_;  
            init_ = true;  
            updata_type_ = LocalMapUpdataType::TIME_UPDATA; 
            // LOG(INFO) << "track init ok"; 
            return;  
        }
        // 位姿预测
        // 判断是否有预测位姿
        if (deltaT.matrix() == Eigen::Isometry3d::Identity().matrix()) {
            curr_pose_ = prev_pose_ * motion_increment_; // 采用tracker自带的匀速运动学模型预测
        } else {
            curr_pose_ = prev_pose_ * deltaT;  
        }
        registration_ptr_->SetInputSource(data.feature_data_); 
        SlamLib::time::TicToc tt;
        registration_ptr_->Solve(curr_pose_);       
        double time = tt.toc("Registration ");
        //std::cout<<"curr_pose_: "<<std::endl<<curr_pose_.matrix()<<std::endl;
        motion_increment_ = prev_pose_.inverse() * curr_pose_;    // 当前帧与上一帧的运动增量
        deltaT = motion_increment_;
        prev_pose_ = curr_pose_;  


        SlamLib::PCLPtr<_PointType> unground_points(new pcl::PointCloud<_PointType>());   // 非地面点 
        unground_points->reserve(data.ori_points_num);
        std::vector< std::vector<uint32_t>> cluster_indices;

        // step: 动态检测
        // 分析本次匹配各个点的匹配质量
        if (dynamic_removal_) {
            tt.tic();   
            dynamic_cloud_->clear();  
            false_dynamic_cloud_->clear();  
            RegistrationResult const& res = registration_ptr_->GetRegistrationResult(); 
            std::vector<uint32_t> candidate_dynamic_cloud_index_1;   // 候选动态点  

            SlamLib::PCLPtr<_PointType> ground_points(new pcl::PointCloud<_PointType>());   // 地面点 
            ground_points->reserve(data.ori_points_num);
            uint32_t point_index = 0; 
            // 遍历所有参与匹配的特征
            for (const auto& feature_res : res) {
                // 遍历该特征的所有的残差  
                for (uint32_t i = 0; i < feature_res.second.residuals_.size(); ++i) {
                    // 地面点剔除
                    if (data.feature_data_.at(feature_res.first)->points[i].type == 3) {
                        ground_points->push_back(data.feature_data_.at(feature_res.first)->points[i]);  
                        continue;
                    };    
                    unground_points->push_back(data.feature_data_.at(feature_res.first)->points[i]);  // 收集所有非地面点
                    // 动态候选点  1、匹配误差较大    2、只考虑一定距离范围内(50m内)
                    if (std::fabs(feature_res.second.residuals_[i]) > 0.2 && 
                            data.feature_data_.at(feature_res.first)->points[i].range < 50 &&
                            data.feature_data_.at(feature_res.first)->points[i].z < 1) {
                        candidate_dynamic_cloud_index_1.push_back(point_index);   // 收集可能的动态点  
                    }
                    ++point_index;   
                }
            }

            std::cout << "candidate_dynamic_cloud_index_1 size: " << candidate_dynamic_cloud_index_1.size() << std::endl;
            
            if (!ref_keyframe_range_img_.empty()) {
                // 可见性检查-用于筛选出部分误检的动态点，它们是由与遮挡、雷达运动而看到新的曲面而生成的新观测
                // 1、将当前帧转换到上一个关键帧坐标下，并生成range image,然后检查当前动态点的可见性
                SlamLib::PCLPtr<_PointType> curr_unground_points_in_ref_keyframe(new pcl::PointCloud<_PointType>());
                Eigen::Isometry3d curr_to_ref_keyframe = ref_keyframe_pose_.inverse() * curr_pose_;  // curr->ref  
                pcl::transformPointCloud (*unground_points, *curr_unground_points_in_ref_keyframe, 
                    curr_to_ref_keyframe.matrix());
                // 更新range值
                updataRange(curr_unground_points_in_ref_keyframe);

                int H = 180 / 1; // 设置图片的高度

                std::vector<uint32_t> candidate_dynamic_cloud_index_2;   // 候选动态点  
                candidate_dynamic_cloud_index_2.reserve(candidate_dynamic_cloud_index_1.size()); 
                std::vector<std::vector<float>> dynamic_range_img(H, std::vector<float>(360, 100.0f)); 
                std::vector<std::pair<int, int>> dynamic_img_pos;
                dynamic_img_pos.reserve(candidate_dynamic_cloud_index_1.size());
                std::vector<int> dynamic_index;
                dynamic_index.reserve(candidate_dynamic_cloud_index_1.size());  

                for (const uint32_t& index : candidate_dynamic_cloud_index_1) {
                    const auto& p = curr_unground_points_in_ref_keyframe->points[index];   
                    float v_angle = std::asin(p.z / p.range);  
                    uint16_t v_index = (H / 2 - 1) - std::floor(v_angle / 0.0175);    
                    float h_angle = -std::atan2(p.y, p.x);
                    if (h_angle < 0) {
                        h_angle += 2 * M_PI;
                    }
                    uint16_t h_index = h_angle / 0.0175;
                    const auto& trans = curr_to_ref_keyframe.translation();

                    int check_patch_H = std::ceil((0.3 / p.range) / 0.0175);     // 垂直方向的移动的步数
                    // int check_patch_W = std::ceil((0.3 / p.range) / 0.0175);     // 水平方向的移动的步数
                    int check_patch_W = check_patch_H; 
                    // dynamic_cloud_->push_back(unground_points->points[index]);  
                    // if (ref_keyframe_range_img_.at<float>(v_index, h_index) < 200.0f) {
                    if (p.range < ref_keyframe_range_img_.at<float>(v_index, h_index) - 0.3) {   
                        // 检测视角，对于LIVOX来说，视角有限，因此之前视角外的点被突然看到也会被误认为是动态点
                        if (view_limited_) {
                            if (h_index < 340 && h_index > 30) 
                                continue;  
                        }
                        // std::cout << "h_index: " << h_index << std::endl;
                        // std::cout << "p.x: " << p.x << ", p.y: " << p.y << std::endl;
                        
                        // std::cout << "cell is 200 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
                        // 存在一些特殊情况，如 
                        //  1、视角接近平行地面点，这些点落在range img的同一个cell中
                        //          但是range却相差巨大，容易被误检为动态点
                        // 2、各种物体的边缘，树、草丛，由与激光采样分辨率较低，也许同一个range img cell中 ref帧的观测在背景上，
                        //         而在curr帧中就采样到了物体上了，这样就产生了range较大的差异，从而被误检为动态点 
                        // 为了避免上述误检，采用一个trick,同时检查该激光点落在的range img cell 附近上下左右的四个cell
                        // 如果附近这几个cell的range值都大于该激光点的值，那么才认为是的动态
                        // 获取上一个cell的观测
                        bool invalid = false; 
                        // 检查该点周围 5 * 5 的patch，这个patch内
                        for (int i = -check_patch_H; i <= check_patch_H; ++i) {
                            if (v_index + i < 0 || v_index + i >= 180) continue;   
                            for (int j = -check_patch_W; j <= check_patch_W; ++j) {
                                if (h_index + j < 0 || h_index + j >= 360) continue;   
                                // if (p.range > ref_keyframe_range_img_.at<float>(v_index + i, h_index + j) - 0.5) {
                                //     invalid = true; 
                                //     // std::cout << "p.range: " << p.range << ", cell: " << ref_keyframe_range_img_.at<float>(v_index + i, h_index + j) << std::endl;
                                //     // std::cout << "i: " << i << ",j: " << j << std::endl;
                                //     break;
                                // }
                                // 如果这个动态点击中的背景属于不可见的位置，那么，如果如果该cell周围一块
                                // 区域都不可见，那么认为该位置附近超过激光的可视范围，该点可能是动态点，
                                // 但是，如果该cell周围区域存在有效的观测，说明当前cell仅仅是处与雷达视野盲区，
                                // 因此，这个点很可能是雷达新看到的点，因此不应该当成动态点
                                if (ref_keyframe_range_img_.at<float>(v_index, h_index) == 100.0f) {
                                    if (ref_keyframe_range_img_.at<float>(v_index + i, h_index + j) < 100.0f ) {
                                        // 如果这个点与车运动方向接近平行，那么认为不可能出现遮挡，因此只能是动态  
                                        // Eigen::Vector2d trans_vec(trans.x(), trans.y());
                                        // trans_vec.normalize();
                                        // Eigen::Vector2d view_vec(p.x, p.y);
                                        // view_vec.normalize();
                                        // float value = trans_vec.dot(view_vec);

                                        // if (std::fabs(value) > 0.99) break;

                                        if (p.range + 0.5 > ref_keyframe_range_img_.at<float>(v_index + i, h_index + j)) {
                                            invalid = true; 
                                            break;
                                        }
                                    }
                                } else {
                                    // 如果参考帧的距离图像cell 有观测值(< 100), 那么如果该cel周围有一个cell的value和当前range
                                    // 接近，说明很可能这是物体边缘
                                    if (std::fabs(p.range - ref_keyframe_range_img_.at<float>(v_index + i, h_index + j)) < 0.5) {
                                        invalid = true; 
                                        break;
                                    }
                                }
                            }
                            if (invalid) break;  
                        }

                        if (invalid) {
                            false_dynamic_cloud_->push_back(unground_points->points[index]);  
                        } else {
                            // dynamic_cloud_->push_back(unground_points->points[index]);  
                            // candidate_dynamic_cloud_index_2.push_back(index);
                            dynamic_range_img[v_index][h_index] = p.range;
                            dynamic_img_pos.push_back(std::make_pair(v_index, h_index));
                            dynamic_index.push_back(index);
                        }
                    } else if (p.range > ref_keyframe_range_img_.at<float>(v_index, h_index) + 0.3 &&
                                        p.range < ref_keyframe_range_img_.at<float>(v_index, h_index) + 20) {
                        // 如果运动向量与动态点的观测向量接近平行，那么这个动态点成立
                        // Eigen::Vector2d trans_vec(trans.x(), trans.y());
                        // trans_vec.normalize();
                        // Eigen::Vector2d view_vec(p.x, p.y);
                        // view_vec.normalize();
                        // float value = trans_vec.dot(view_vec);
                        // if (std::fabs(value) > 0.99) {
                        //     bool invalid = false; 
                        //     // 检查该点周围 5 * 5 的patch，这个patch内每个cell的range都要小与当前range
                        //     // 避免平行，边缘的误检
                        //     for (int i = -check_patch_H; i <= check_patch_H; ++i) {
                        //         if (v_index + i < 0 || v_index + i >= 180) continue;   
                        //         for (int j = -check_patch_W; j <= check_patch_W; ++j) {
                        //             if (h_index + j < 0 || h_index + j >= 360) continue;   

                        //             if (ref_keyframe_range_img_.at<float>(v_index + i, h_index + j) < 100.0f ) {
                        //                 if (p.range < ref_keyframe_range_img_.at<float>(v_index + i, h_index + j) + 0.3) {
                        //                     invalid = true; 
                        //                     break;
                        //                 }
                        //             }
                        //         }
                        //         if (invalid) {
                        //             break; 
                        //         }
                        //     }
                            
                        //     if (invalid) {
                        //         // false_dynamic_cloud_->push_back(unground_points->points[index]);  
                        //     } else {
                        //         // dynamic_cloud_->push_back(unground_points->points[index]);  
                        //         // candidate_dynamic_cloud_index_2.push_back(index);
                        //         // dynamic_range_img[v_index][h_index] = p.range;
                        //         // dynamic_img_pos.push_back(std::make_pair(v_index, h_index));
                        //         // dynamic_index.push_back(index);
                        //     }
                        // }
                    }
                }

                // box滤波
                for (int n = 0; n < dynamic_img_pos.size(); ++n) {
                    uint8_t score = 0;
                    const auto& range = dynamic_range_img[dynamic_img_pos[n].first][dynamic_img_pos[n].second];
                    int H = std::ceil((0.5 / range) / 0.0175);     // 垂直方向的移动的步数
                    int W = std::ceil((0.5 / range) / 0.0175);     // 水平方向的移动的步数
                    // std::cout << "range: " << range << ", H: " << H << ", W: " << W << std::endl;

                    for (int i = -H; i <= H; ++i) {
                        if (i == 0) continue;  
                        if (dynamic_img_pos[n].first + i < 0 || dynamic_img_pos[n].first + i >= 180) continue; 
                        for (int j = -W; j <= W; ++j) {
                            if (dynamic_img_pos[n].second + j < 0 || dynamic_img_pos[n].second + j >= 360)
                                continue;  
                            if (dynamic_range_img[dynamic_img_pos[n].first + i][dynamic_img_pos[n].second + j] == 200) 
                                continue;
                            if (std::fabs(dynamic_range_img[dynamic_img_pos[n].first + i][dynamic_img_pos[n].second + j]
                                    - range) < 0.5) {
                                ++score;  
                            }
                        }
                    }
                    if (score) {
                        candidate_dynamic_cloud_index_2.push_back(dynamic_index[n]);
                        dynamic_cloud_->push_back(unground_points->points[dynamic_index[n]]);  
                    } 
                    // else {
                    //     std::cout << "range: " << range << "H: " << H << ",W: " << W << std::endl;
                    // }
                }

                std::cout << "candidate_dynamic_cloud_index_2 size: " << candidate_dynamic_cloud_index_2.size() << std::endl;
                // 对候选动态点进行区域生长
                typename SlamLib::pointcloud::EuclideanCluster<_PointType>::Option option;
                SlamLib::pointcloud::EuclideanCluster<_PointType> cluster(option);
                cluster.RegionGrowing(unground_points, candidate_dynamic_cloud_index_2, cluster_indices);  

                // dynamic_cloud_->clear();
                // for (const auto& cluster : cluster_indices ) {
                //     for (const auto& i : cluster) {
                //         dynamic_cloud_->push_back(unground_points->points[i]);
                //     }
                // }
            }

            tt.toc("dynamic detect ");
        }
    
        // 判定是否需要更新local map  
        updata_type_ = needUpdataLocalMap(curr_pose_, data.timestamp_start_);  
        // 判定是否需要更新localmap
        // 若需要更新local map  1、读取匹配信息，结合匹配信息更新Local map   2、todo: 去除动态目标点云 
        if (updata_type_ != LocalMapUpdataType::NO_UPDATA) {
            ref_keyframe_pose_ = last_keyframe_pose_;  
            last_keyframe_pose_ = curr_pose_; // 更新关键帧Pose 
            last_keyframe_time_ = data.timestamp_start_;   

            if (dynamic_removal_) {
                SlamLib::PCLPtr<_PointType> static_pointcloud(new pcl::PointCloud<_PointType>);  
                // 构造出静态点云 
                std::vector<bool> cluster_flags(unground_points->points.size(), false);
                for (const auto& cluster : cluster_indices ) {
                    for (const auto& i : cluster) {
                        cluster_flags[i] = true;  
                    }
                }
                // dynamic_cloud_->clear();
                for (int i = 0; i < cluster_flags.size(); ++i) {
                    if (!cluster_flags[i]) {
                        static_pointcloud->push_back(unground_points->points[i]);  
                    } else {
                        // dynamic_cloud_->push_back(unground_points->points[i]);
                    }
                }

                SlamLib::PCLPtr<_PointType> ref_keyframe_unground_cloud = last_keyframe_unground_cloud_; 
                // last_keyframe_unground_cloud_ = unground_points;    // 原端的地面点可能会遮挡物体点
                // last_keyframe_unground_cloud_ = data.feature_data_.at("filtered");  
                last_keyframe_unground_cloud_ = static_pointcloud;  
                // *static_pointcloud += *ground_points;  

                if (ref_keyframe_unground_cloud != nullptr) {
                    // 对ref_keyframe_unground_cloud_进行进一步滤除
                    SlamLib::PCLPtr<_PointType> ref_unground_points_in_last_keyframe(new pcl::PointCloud<_PointType>());
                    Eigen::Isometry3d ref_to_last_keyframe = last_keyframe_pose_.inverse() * ref_keyframe_pose_;  // ref->curr  
                    pcl::transformPointCloud (*ref_keyframe_unground_cloud, *ref_unground_points_in_last_keyframe, 
                        ref_to_last_keyframe.matrix());
                    // 更新range值
                    updataRange(ref_unground_points_in_last_keyframe);
                    cv::Mat ref_unground_points_in_last_keyframe_img_ = generateRangeImage(ref_unground_points_in_last_keyframe, "_ref"); 
                    cv::Mat last_keyframe_unground_cloud_img_ = generateRangeImage(last_keyframe_unground_cloud_, "_last"); 
                    // 进行对比，如果last图像某个cell的range 大于 ref图像同一个cell的range (即last帧的激光穿过了ref帧的激光)
                    if (view_limited_) {             
                        int H = 180 / 1; // 设置图片的高度    +-90 视野
                        int W = 360 / 1; // 设置图片的宽度           
                        cv::Mat dyna_flag_img(H, W, CV_8U, cv::Scalar(0));
                        
                        for (int v = 0; v < 180; ++v) {
                            for (int h = 0; h < 30; ++h) {
                                if (ref_unground_points_in_last_keyframe_img_.at<float>(v, h) == 100.0f
                                        || last_keyframe_unground_cloud_img_.at<float>(v, h) == 100.0f) {
                                    continue;
                                }
                                if (last_keyframe_unground_cloud_img_.at<float>(v, h) 
                                        > ref_unground_points_in_last_keyframe_img_.at<float>(v, h) + 1) {
                                    // std::cout << "diff: " << last_keyframe_unground_cloud_img_.at<float>(v, h) 
                                    //     - ref_unground_points_in_last_keyframe_img_.at<float>(v, h) << std::endl;
                                    dyna_flag_img.at<uint8_t>(v, h) = 1;   // 标记  
                                }
                            }
                        }

                        for (int v = 0; v < 180; ++v) {
                            for (int h = 340; h < 360; ++h) {
                                if (ref_unground_points_in_last_keyframe_img_.at<float>(v, h) == 100.0f
                                        || last_keyframe_unground_cloud_img_.at<float>(v, h) == 100.0f) {
                                    continue;
                                }
                                if (last_keyframe_unground_cloud_img_.at<float>(v, h) 
                                        > ref_unground_points_in_last_keyframe_img_.at<float>(v, h) + 1) {
                                    dyna_flag_img.at<uint8_t>(v, h) = 1;   // 标记  
                                }
                            }
                        }
                        SlamLib::PCLPtr<_PointType> good_ref_unground_points_in_last_keyframe(new pcl::PointCloud<_PointType>());
                        good_ref_unground_points_in_last_keyframe->reserve(ref_unground_points_in_last_keyframe->size());
                        // 点云选择
                        for (const auto& p : *ref_unground_points_in_last_keyframe) {
                            float v_angle = std::asin(p.z  / p.range);  
                            uint16_t v_index = (H / 2 - 1) - std::floor(v_angle / 0.0175);
                            float h_angle = -std::atan2(p.y, p.x);
                            if (h_angle < 0) {
                                h_angle += 2 * M_PI;
                            }
                            uint16_t h_index = h_angle / 0.0175;
                            // 图像每个cell 取落在该cell中的最近的range  
                            if (dyna_flag_img.at<uint8_t>(v_index, h_index)) {
                                // false_dynamic_cloud_->push_back(p);  
                            } else {
                                good_ref_unground_points_in_last_keyframe->push_back(p);
                            }
                        }
                        // 将good_ref_unground_points_in_last_keyframe 转换到ref系下
                        pcl::transformPointCloud (*good_ref_unground_points_in_last_keyframe, *ref_keyframe_unground_cloud, 
                            ref_to_last_keyframe.inverse().matrix());
                    }
                    // 修改局部地图
                    // 转换到map系下
                    SlamLib::PCLPtr<_PointType> ref_unground_points_in_map(new pcl::PointCloud<_PointType>());
                    pcl::transformPointCloud (*ref_keyframe_unground_cloud, *ref_unground_points_in_map, 
                            ref_keyframe_pose_.matrix());
                    local_map_->AmendData("filtered", local_map_->GetWindowSize("filtered") - 1, ref_unground_points_in_map);
                    ref_keyframe_range_img_ = generateRangeImage(ref_keyframe_unground_cloud); 
                }

                SlamLib::time::TicToc tt;
                SlamLib::FeaturePointCloudContainer<_PointType> static_pointcloud_container;    
                static_pointcloud_container.insert(std::make_pair("filtered", static_pointcloud));  
                updateLocalMap(static_pointcloud_container, 
                    curr_pose_, updata_type_); 
                tt.toc("updateLocalMap ");
            } else {
                updateLocalMap(data.feature_data_, curr_pose_, updata_type_); // 更新localmap     kdtree: 9ms 左右 
            }

            return;
        }
        // tt.toc("done: ");  
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    const SlamLib::PCLPtr<_PointType>& GetDynamicCloud() const {
        return dynamic_cloud_; 
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    const SlamLib::PCLPtr<_PointType>& GetFalseDynamicCloud() const {
        return false_dynamic_cloud_; 
    }
    
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    const typename LidarTrackerBase<_PointType>::LocalMapContainer& 
    GetLocalMap() const {
        return local_map_->GetLocalMap();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief: 获取在local map 坐标系的当前坐标 
     */            
    Eigen::Isometry3d const& GetCurrPoseInLocalFrame() const override {
        return curr_pose_;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    bool HasUpdataLocalMap() const override {
        if (updata_type_ != LocalMapUpdataType::NO_UPDATA) {
            return true;
        }
        return false;  
    }

    void ResetLocalmap() override {
        local_map_->Reset();  
        init_ = false;  
    }

protected:

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief 
     *  
     * @param points 
     */
    cv::Mat generateRangeImage(const SlamLib::PCLConstPtr<_PointType>& points, const std::string& label = "0") {
        // 图片垂直范围：+-90度   分辨率 2度
        // 图片水平范围：0-360度，分辨率   1度  
        int H = 180 / 1; // 设置图片的高度    +-90 视野
        int W = 360 / 1; // 设置图片的宽度
        cv::Mat range_img(H, W, CV_32F, cv::Scalar(100.0f));
        for (const auto& p : *points) {
            float v_angle = std::asin(p.z / p.range);  
            uint16_t v_index = (H / 2 - 1) - std::floor(v_angle / 0.0175);
            float h_angle = -std::atan2(p.y, p.x);
            if (h_angle < 0) {
                h_angle += 2 * M_PI;
            }
            uint16_t h_index = h_angle / 0.0175;
            // 图像每个cell 取落在该cell中的最近的range  
            if (p.range < range_img.at<float>(v_index, h_index)) {
                range_img.at<float>(v_index, h_index) = p.range;  
            }
            // if (h_index > 300) {
                // range_img.at<float>(v_index, h_index) = 10.0f;
                // std::cout << "h_index: " << h_index << std::endl;
            // }
        }

        // for (int i = 0; i < 100; ++i) {
        //     range_img.at<float>(60, i) = 0;
        // }

        cv::Mat gray_img;
        cv::normalize(range_img, gray_img, 0, 255, cv::NORM_MINMAX, CV_8U);
        // cv::Mat resized_mat;
        // cv::resize(gray_img, resized_mat, cv::Size(1000, 1000), 0, 0, cv::INTER_LINEAR);
        cv::imshow("range_img_" + label, gray_img);
        cv::waitKey(10);  
        return range_img; 
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief 获取投影在距离图像上的水平坐标
     * 
     * @return uint16_t 
     */
    uint16_t getRangeImageHorizonIndex(const _PointType& p) {
        float h_angle = -std::atan2(p.y, p.x);
        if (h_angle < 0) {
            h_angle += 2 * M_PI;
        }
        return h_angle / rangeImg_opt_.h_angle_res_rad_;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief 获取投影在距离图像上的垂直坐标
     * 
     * @return uint16_t 
     */
    uint16_t getRangeImageVerticalIndex(const _PointType& p) {
        float v_angle = std::asin(p.z / p.range);  
        return (rangeImg_opt_.h_size / 2 - 1) - std::floor(v_angle / rangeImg_opt_.v_angle_res_rad_);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief 
     * 
     * @param points 
     */
    void updataRange(const SlamLib::PCLPtr<_PointType>& points) {
        for (auto& p : *points) {
            p.range = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief: 更新局部地图 
     * @param curr_points 用于更新local map 的点云数据  
     * @param info 当前点云的匹配信息  
     * @param T 相对于local map坐标系的变换矩阵
     * @param updata_type local map的更新方式 
     */            
    void updateLocalMap(SlamLib::FeaturePointCloudContainer<_PointType> const& curr_points, 
            Eigen::Isometry3d const& T, LocalMapUpdataType const& updata_type) {
        // 变换点云到odom坐标 
        SlamLib::PCLPtr<_PointType> transformed_cloud(new pcl::PointCloud<_PointType>());
        // 遍历全部特征的点云数据 pointcloud_data_ 
        for (auto iter = curr_points.begin(); iter != curr_points.end(); ++iter) {
            if (iter->second->empty()) continue;   // 判断数据是否为空
            // 更新地图
            pcl::transformPointCloud (*(iter->second), *transformed_cloud, T.matrix());

            if (updata_type == LocalMapUpdataType::MOTION_UPDATA) {   
                local_map_->UpdateLocalMapForMotion(iter->first, transformed_cloud);  
            } else if (updata_type == LocalMapUpdataType::TIME_UPDATA) {
                local_map_->UpdateLocalMapForTime(iter->first, transformed_cloud);  
            }
        }
        // 将更新后的地图设置为匹配的local map 
        registration_ptr_->SetInputTarget(local_map_);     
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief: 检查是否需要更新局部地图
     * @details:  1、检查运动是否足够(local map 没满时，阈值低一些)    2、时间是否足够久 10s  
     */            
    LocalMapUpdataType needUpdataLocalMap(
        Eigen::Isometry3d const& curr_pose, double const& curr_time) {
        // 检查时间
        if (curr_time - last_keyframe_time_ > option_.TIME_INTERVAL_) {
            return LocalMapUpdataType::TIME_UPDATA;
        } 
        // 检查运动
        // 求出相对于上一个关键帧的变换
        Eigen::Isometry3d delta_transform = last_keyframe_pose_.inverse() * curr_pose;
        double delta_translation = delta_transform.translation().norm();         
        // 旋转矩阵对应 u*theta  对应四元数  e^u*theta/2  = [cos(theta/2), usin(theta/2)]
        Eigen::Quaterniond q_trans(delta_transform.rotation());
        q_trans.normalize();   
        double delta_angle = std::acos(q_trans.w()) * 2;     // 获得弧度    45度 约等于 0.8  
        // 满足关键帧条件
        if (delta_translation > option_.THRESHOLD_TRANS_ 
            || delta_angle > option_.THRESHOLD_ROT_) {
            //std::cout<<common::GREEN<<"NEED MOTION_UPDATA!"<<common::RESET<<std::endl;
            return LocalMapUpdataType::MOTION_UPDATA;
        }
        return LocalMapUpdataType::NO_UPDATA;  
    }

private:
    Option option_; 
    rangeImageOption rangeImg_opt_;  
    bool init_;   
    bool local_map_full_ = false;  
    bool dynamic_removal_ = false;  
    bool view_limited_ = false;  
    // 匹配算法 
    // 输入源点云类型：LocalMapInput          输入目标点云类型：FeatureInfo<_PointType>
    typename SlamLib::pointcloud::RegistrationBase<_PointType>::Ptr registration_ptr_;
    LocalMapPtr local_map_;  
    SlamLib::PCLPtr<_PointType> dynamic_cloud_;
    SlamLib::PCLPtr<_PointType> false_dynamic_cloud_;
    SlamLib::PCLPtr<_PointType> last_keyframe_unground_cloud_ = nullptr;
    cv::Mat ref_keyframe_range_img_; 
    Eigen::Isometry3d prev_pose_;  // 上一帧的位姿
    Eigen::Isometry3d curr_pose_;   // 上一帧的位姿
    Eigen::Isometry3d predict_trans_;  // 预测位姿
    Eigen::Isometry3d motion_increment_;  // 运动增量 
    Eigen::Isometry3d last_keyframe_pose_;  // 上一个关键帧的位姿
    Eigen::Isometry3d ref_keyframe_pose_;  // 上一个的上一个关键帧的位姿
    double last_keyframe_time_;  // 上一个关键帧的位姿
    LocalMapUpdataType updata_type_; 
}; // class LidarTracker
} // namespace 
