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
    // using LocalMapConstPtr =  typename SlamLib::map::PointCloudLocalMapBase<_PointType>::ConstPtr;
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
        } catch (const YAML::Exception& e) {
            LOG(ERROR) << "YAML Exception: " << e.what();
        }

        dynamic_cloud_.reset(new pcl::PointCloud<_PointType>());
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
        // static double sum = 0;
        // static int i = 0;
        // if (i < 500) {
        //     sum += time;
        //     i++;
        // } else {
        //     LOG(INFO) << "avg time:"<<sum / i; 
        // }
        //std::cout<<"curr_pose_: "<<std::endl<<curr_pose_.matrix()<<std::endl;
        motion_increment_ = prev_pose_.inverse() * curr_pose_;    // 当前帧与上一帧的运动增量
        deltaT = motion_increment_;
        prev_pose_ = curr_pose_;  

        
        // step: 动态检测
        // 分析本次匹配各个点的匹配质量
        dynamic_cloud_->clear();  
        RegistrationResult const& res = registration_ptr_->GetRegistrationResult(); 
        std::vector<uint32_t> candidate_dynamic_cloud_index_1;   // 候选动态点  
        SlamLib::PCLPtr<_PointType> unground_points(new pcl::PointCloud<_PointType>());   // 非地面点 
        unground_points->reserve(data.ori_points_num);
        uint32_t point_index = 0; 
        // 遍历所有参与匹配的特征
        for (const auto& feature_res : res) {
            for (uint32_t i = 0; i < feature_res.second.residuals_.size(); ++i) {
                if (data.feature_data_.at(feature_res.first)->points[i].type == 3) continue;    // 地面点剔除
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

        if (ref_keyframe_unground_cloud_ != nullptr) {
            // 可见性检查-用于筛选出部分误检的动态点，它们是由与遮挡、雷达运动而看到新的曲面而生成的新观测
            // 1、将当前帧转换到上一个关键帧坐标下，并生成range image,然后检查当前动态点的可见性
            SlamLib::PCLPtr<_PointType> curr_unground_points_in_ref_keyframe(new pcl::PointCloud<_PointType>());
            Eigen::Isometry3d curr_to_ref_keyframe = ref_keyframe_pose_.inverse() * curr_pose_;  
            pcl::transformPointCloud (*unground_points, *curr_unground_points_in_ref_keyframe, 
                curr_to_ref_keyframe.matrix());
            // 更新range值
            updataRange(curr_unground_points_in_ref_keyframe);
            // 生成range image
            // cv::Mat curr_range_img = generateRangeImage(curr_unground_points_in_ref_keyframe); 

            // std::vector<uint32_t> candidate_dynamic_cloud_index_2;   // 进一步的候选点


            int H = 180 / 10; // 设置图片的高度

            // for (const uint32_t& index : candidate_dynamic_cloud_index_1) {
            //     const auto& p = curr_unground_points_in_ref_keyframe->points[index];   
            //     float v_angle = std::asin(p.z / p.range);  
            //     uint16_t v_index = (H / 2 - 1) - std::floor(v_angle / 0.1745);
            //     float h_angle = -std::atan2(p.y, p.x);
            //     if (h_angle < 0) {
            //         h_angle += 2 * M_PI;
            //     }
            //     uint16_t h_index = h_angle / 0.0873;

            //     if (p.range == curr_range_img.at<float>(v_index, h_index)) {
            //         candidate_dynamic_cloud_index_2.push_back(index);  
            //     }
            // }

            // std::cout << "candidate_dynamic_cloud_index size: " << candidate_dynamic_cloud_index_2.size() << std::endl;

            cv::Mat ref_keyframe_range_img = generateRangeImage(ref_keyframe_unground_cloud_); 

            // for (const uint32_t& index : candidate_dynamic_cloud_index_2) {
            //     const auto& p = curr_unground_points_in_ref_keyframe->points[index];   
            //     float v_angle = std::asin(p.z / p.range);  
            //     uint16_t v_index = (H / 2 - 1) - std::floor(v_angle / 0.1745);
            //     float h_angle = -std::atan2(p.y, p.x);
            //     if (h_angle < 0) {
            //         h_angle += 2 * M_PI;
            //     }
            //     uint16_t h_index = h_angle / 0.0873;

            //     if (std::fabs(p.range - ref_keyframe_range_img.at<float>(v_index, h_index)) > 1) {
            //         if (ref_keyframe_range_img.at<float>(v_index, h_index) < 200.0f) {
            //             // std::cout << "cell is 200 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
            //             dynamic_cloud_->push_back(unground_points->points[index]);  
            //         }
            //     }
            // }

                for (const uint32_t& index : candidate_dynamic_cloud_index_1) {
                const auto& p = curr_unground_points_in_ref_keyframe->points[index];   
                float v_angle = std::asin(p.z / p.range);  
                uint16_t v_index = (H / 2 - 1) - std::floor(v_angle / 0.1745);
                float h_angle = -std::atan2(p.y, p.x);
                if (h_angle < 0) {
                    h_angle += 2 * M_PI;
                }
                uint16_t h_index = h_angle / 0.0524;

                if (ref_keyframe_range_img.at<float>(v_index, h_index) < 200.0f) {
                    if (p.range < ref_keyframe_range_img.at<float>(v_index, h_index) - 1) {   
                        // std::cout << "cell is 200 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
                        dynamic_cloud_->push_back(unground_points->points[index]);  
                        std::cout << "p.range: " << p.range << ", ref_keyframe_range_img : " << ref_keyframe_range_img.at<float>(v_index, h_index) << std::endl;
                    }
                }
            }

            std::cout << "dynamic_cloud_ size: " << dynamic_cloud_->size() << std::endl;

        }





        // 判定是否需要更新local map  
        updata_type_ = needUpdataLocalMap(curr_pose_, data.timestamp_start_);  
        // 判定是否需要更新localmap
        // 若需要更新local map  1、读取匹配信息，结合匹配信息更新Local map   2、todo: 去除动态目标点云 
        if (updata_type_ != LocalMapUpdataType::NO_UPDATA) {
            ref_keyframe_pose_ = last_keyframe_pose_;  
            last_keyframe_pose_ = curr_pose_; // 更新关键帧Pose 
            last_keyframe_time_ = data.timestamp_start_;   
            ref_keyframe_unground_cloud_ = last_keyframe_unground_cloud_; 
            last_keyframe_unground_cloud_ = unground_points;  
            SlamLib::time::TicToc tt;
            updateLocalMap(data.feature_data_, curr_pose_, updata_type_); // 更新localmap     kdtree: 9ms 左右 
            tt.toc("updateLocalMap ");

            return;
        }
        // tt.toc("done: ");  
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    const SlamLib::PCLPtr<_PointType>& GetDynamicCloud() const {
        return dynamic_cloud_; 
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

protected:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief 
     *  
     * @param points 
     */
    cv::Mat generateRangeImage(const SlamLib::PCLConstPtr<_PointType>& points) {
        // 图片垂直范围：+-90度   分辨率 2度
        // 图片水平范围：0-360度，分辨率   1度  
        int H = 180 / 10; // 设置图片的高度    +-90 视野
        int W = 360 / 3; // 设置图片的宽度
        cv::Mat range_img(H, W, CV_32F, cv::Scalar(200.0f));
        for (const auto& p : *points) {
            float v_angle = std::asin(p.z / p.range);  
            uint16_t v_index = (H / 2 - 1) - std::floor(v_angle / 0.1745);
            float h_angle = -std::atan2(p.y, p.x);
            if (h_angle < 0) {
                h_angle += 2 * M_PI;
            }
            uint16_t h_index = h_angle / 0.0524;
            if (p.range < range_img.at<float>(v_index, h_index)) {
                range_img.at<float>(v_index, h_index) = p.range;  
            }
        }

        cv::Mat gray_img;
        cv::normalize(range_img, gray_img, 0, 255, cv::NORM_MINMAX, CV_8U);
        // cv::Mat resized_mat;
        // cv::resize(gray_img, resized_mat, cv::Size(1000, 1000), 0, 0, cv::INTER_LINEAR);
        cv::imshow("range_img", gray_img);
        cv::waitKey(10);  
        return range_img; 
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
        double delta_angle = std::acos(q_trans.w())*2;     // 获得弧度    45度 约等于 0.8  
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
    bool init_;   
    bool local_map_full_ = false;  
    // 匹配算法 
    // 输入源点云类型：LocalMapInput          输入目标点云类型：FeatureInfo<_PointType>
    typename SlamLib::pointcloud::RegistrationBase<_PointType>::Ptr registration_ptr_;
    LocalMapPtr local_map_;  
    SlamLib::PCLPtr<_PointType> dynamic_cloud_;
    SlamLib::PCLPtr<_PointType> last_keyframe_unground_cloud_ = nullptr;
    SlamLib::PCLPtr<_PointType> ref_keyframe_unground_cloud_ = nullptr;     // 上一个的上一个关键帧的非地面点 
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
