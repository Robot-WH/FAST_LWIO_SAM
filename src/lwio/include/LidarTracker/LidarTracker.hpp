#pragma once 

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <yaml-cpp/yaml.h>
#include "LidarTrackerBase.hpp"
#include "Map/LocalMap.hpp"
#include "Map/TimedSlidingLocalMap.hpp"
#include "Map/iVoxLocalMap.hpp"
#include "Map/iVoxTimedSlidingLocalMap.hpp"
#include "Algorithm/PointClouds/registration/ceres_edgeSurfFeatureRegistration.hpp"
#include "Algorithm/PointClouds/registration/edgeSurfFeatureRegistration.hpp"
#include "tic_toc.h"
#include "Common/color.hpp"

namespace Slam3D {
namespace {

using namespace Algorithm; 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: tracker LoclMap匹配模式框架
 * @details:  适应多种Local Map构造方式 ， 以及多种匹配方式  a. 特征ICP  b. NDT/GICP
 * @param _PointType 使用的特征点的类型  
 */    
template<typename _PointType>
class LidarTracker : public LidarTrackerBase<_PointType> {
    private:
        using PointCloudPtr = typename pcl::PointCloud<_PointType>::Ptr;  
        using PointCloudConstPtr = typename pcl::PointCloud<_PointType>::ConstPtr;  
        using LocalMapConstPtr =  typename PointCloudLocalMapBase<_PointType>::ConstPtr;
        using RegistrationResult = typename RegistrationBase<_PointType>::RegistrationResult; 
        using PointVector = typename RegistrationBase<_PointType>::PointVector;  
        enum class LocalMapUpdataType {
            NO_UPDATA = 0,
            MOTION_UPDATA,
            TIME_UPDATA
        };
    public:
        using LocalMapContainer = std::unordered_map<std::string, PointCloudConstPtr>;     // Local map 类型 
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
                    registration_ptr_.reset(new CeresEdgeSurfFeatureRegistration<_PointType>("", point_label)); 
                } else if (solve == "LM" || solve == "GN") {
                    typename EdgeSurfFeatureRegistration<_PointType>::Option option;
                    option.surf_label_ = point_label;
                    option.max_iterater_count_ = 
                        yaml["tracker"]["registration"]["point_plane_icp"]["max_iterater_count"].as<int>();
                    option.norm_iterater_count_ = 
                        yaml["tracker"]["registration"]["point_plane_icp"]["norm_iterater_count"].as<int>();
                    if (solve == "GN") {
                        option.method_ = EdgeSurfFeatureRegistration<_PointType>::OptimizeMethod::GN;
                        option.gn_option_.max_iterater_count_ = 
                            yaml["tracker"]["registration"]["point_plane_icp"]["GN"]["max_iterater_count"].as<int>();
                    } else {
                        option.method_ = EdgeSurfFeatureRegistration<_PointType>::OptimizeMethod::LM;
                        option.lm_option_.max_iterater_count_ = 
                            yaml["tracker"]["registration"]["point_plane_icp"]["LM"]["max_iterater_count"].as<int>();
                    }
                    registration_ptr_.reset(new EdgeSurfFeatureRegistration<_PointType>(option)); 
                }
            } else if (registration_method == "ndt") {
            } else if (registration_method == "gicp") {
            }
            // 设置Local map 
            std::string localmap_method = yaml["tracker"]["localmap"]["method"].as<std::string>();
            LOG(INFO) << "localmap_method:"<<localmap_method; 
            if (localmap_method == "time_sliding") {
                typename TimedSlidingLocalMap<_PointType>::Option option;  
                option.window_size_ = yaml["tracker"]["localmap"]["time_sliding"]["window_size"].as<int>();
                option.use_kdtree_search_ = yaml["tracker"]["localmap"]["time_sliding"]["kdtree_enable"].as<bool>();
                local_map_.reset(new TimedSlidingLocalMap<_PointType>(option));
            } else if (localmap_method == "area_sliding") {
            } else if (localmap_method == "ivox") {
                typename iVoxLocalMap<_PointType>::Option option; 
                option.downsampling_size_ = yaml["tracker"]["localmap"]["ivox"]["downsampling_size"].as<double>();
                LOG(INFO) << "ivox map downsampling_size:"<<option.downsampling_size_; 
                option.ivox_option_.resolution_ = yaml["tracker"]["localmap"]["ivox"]["resolution"].as<double>();
                LOG(INFO) << "ivox resolution:"<<option.ivox_option_.resolution_; 
                option.ivox_option_.capacity_ = yaml["tracker"]["localmap"]["ivox"]["capacity"].as<int>();
                LOG(INFO) << "ivox capacity:"<<option.ivox_option_.capacity_; 
                int nearby_type = yaml["tracker"]["localmap"]["ivox"]["nearby_type"].as<int>();
                if (nearby_type == 18) {
                    LOG(INFO) << "ivox nearby_type: NEARBY18"; 
                    option.ivox_option_.nearby_type_ = iVoxLocalMap<_PointType>::IvoxNearbyType::NEARBY18;
                }
                local_map_.reset(new iVoxLocalMap<_PointType>(option));
            } else if (localmap_method == "timedIvox") {
                typename IvoxTimedSlidingLocalMap<_PointType>::Option option; 
                option.window_size_ = yaml["tracker"]["localmap"]["timedIvox"]["window_size"].as<float>();
                LOG(INFO) << "timedIvox map window_size:"<<option.window_size_; 
                option.ivox_option_.resolution_ = yaml["tracker"]["localmap"]["timedIvox"]["resolution"].as<float>();
                LOG(INFO) << "ivox resolution:"<<option.ivox_option_.resolution_; 
                option.ivox_option_.capacity_ = yaml["tracker"]["localmap"]["timedIvox"]["capacity"].as<int>();
                LOG(INFO) << "ivox capacity:"<<option.ivox_option_.capacity_; 
                int nearby_type = yaml["tracker"]["localmap"]["timedIvox"]["nearby_type"].as<int>();
                if (nearby_type == 18) {
                    LOG(INFO) << "ivox nearby_type: NEARBY18"; 
                    option.ivox_option_.nearby_type_ = iVoxLocalMap<_PointType>::IvoxNearbyType::NEARBY18;
                }
                local_map_.reset(new IvoxTimedSlidingLocalMap<_PointType>(option));
            }
            // 设置关键帧提取参数
            option_.THRESHOLD_TRANS_ = yaml["tracker"]["keyframe_update"]["translation"].as<float>();
            LOG(INFO) << "local map update translation:"<<option_.THRESHOLD_TRANS_;
            option_.THRESHOLD_ROT_ = yaml["tracker"]["keyframe_update"]["rotation"].as<float>();
            LOG(INFO) << "local map update rotation:"<<option_.THRESHOLD_ROT_;
            option_.TIME_INTERVAL_ = yaml["tracker"]["keyframe_update"]["time"].as<float>();
            LOG(INFO) << "local map update time:"<<option_.TIME_INTERVAL_;

            updata_type_ = LocalMapUpdataType::NO_UPDATA; 
        }

        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 激光tracker 求解 
         * @details:  求解出当前激光与上一帧激光数据的增量  
         * @param[in] data 特征点云数据
         * @param[in] timestamp 时间戳  
         * @param[out] deltaT 输出的增量
         */            
        virtual void Solve(CloudContainer<_PointType> const& data, 
                Eigen::Isometry3d &deltaT) override {
            // local map 初始化
            if (init_ == false) {   
                updateLocalMap(data.pointcloud_data_, Eigen::Isometry3d::Identity(), 
                    LocalMapUpdataType::TIME_UPDATA);
                curr_pose_ = Eigen::Isometry3d::Identity();  
                prev_pose_ = Eigen::Isometry3d::Identity();  
                motion_increment_ = Eigen::Isometry3d::Identity();  
                last_keyframe_pose_ = Eigen::Isometry3d::Identity();  
                last_keyframe_time_ = data.time_stamp_;  
                init_ = true;  
                updata_type_ = LocalMapUpdataType::TIME_UPDATA; 
                LOG(INFO) << "track init ok"; 
                return;  
            }
            // 位姿预测
            // 判断是否有预测位姿
            if (deltaT.matrix() == Eigen::Isometry3d::Identity().matrix()) {
                curr_pose_ = prev_pose_ * motion_increment_; // 采用匀速运动学模型预测
            } else {
                curr_pose_ = prev_pose_ * deltaT;  
            }
            registration_ptr_->SetInputSource(data.pointcloud_data_); 
            TicToc tt;
            registration_ptr_->Solve(curr_pose_);       
            double time = tt.toc("Registration ");
            static double sum = 0;
            static int i = 0;
            if (i < 500) {
                sum += time;
                i++;
            } else {
                LOG(INFO) << "avg time:"<<sum / i; 
            }
            //std::cout<<"curr_pose_: "<<std::endl<<curr_pose_.matrix()<<std::endl;
            motion_increment_ = prev_pose_.inverse() * curr_pose_;    // 当前帧与上一帧的运动增量
            deltaT = motion_increment_;
            prev_pose_ = curr_pose_;  
            // 判定是否需要更新local map  
            updata_type_ = needUpdataLocalMap(curr_pose_, data.time_stamp_);  
            // 判定是否需要更新localmap
            // 若需要更新local map  1、读取匹配信息，结合匹配信息更新Local map   2、todo: 去除动态目标点云 
            if (updata_type_ != LocalMapUpdataType::NO_UPDATA) {
                last_keyframe_pose_ = curr_pose_; // 更新关键帧Pose 
                last_keyframe_time_ = data.time_stamp_;   
                TicToc tt;
                updateLocalMap(data.pointcloud_data_, curr_pose_, updata_type_); // 更新localmap     kdtree: 9ms 左右 
                tt.toc("updateLocalMap ");
                return;
            }
            // tt.toc("done: ");  
        }
        
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        LocalMapContainer GetLocalMap() {
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
        bool HasUpdataLocalMap() {
            if (updata_type_ != LocalMapUpdataType::NO_UPDATA) {
                return true;
            }
            return false;  
        }

    protected:
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 更新局部地图 
         * @param curr_points 用于更新local map 的点云数据  
         * @param info 当前点云的匹配信息  
         * @param T 相对于local map坐标系的变换矩阵
         * @param updata_type local map的更新方式 
         */            
        void updateLocalMap(FeaturePointCloudContainer<_PointType> const& curr_points, 
                Eigen::Isometry3d const& T, LocalMapUpdataType const& updata_type) {
            // 变换点云到odom坐标 
            PointCloudPtr transformed_cloud(new pcl::PointCloud<_PointType>());
            // 读取匹配的结果(每个点的残差)
            RegistrationResult const& res = registration_ptr_->GetRegistrationResult(); 
            // 遍历全部特征的点云数据 pointcloud_data_ 
            for (auto iter = curr_points.begin(); iter != curr_points.end(); ++iter) {
                if (iter->second->empty()) continue;   // 判断数据是否为空
                // 更新地图
                pcl::transformPointCloud (*(iter->second), *transformed_cloud, T.matrix());
                if (updata_type == LocalMapUpdataType::MOTION_UPDATA) {   
                    local_map_->UpdateLocalMapForMotion(iter->first, transformed_cloud, res.at(iter->first).nearly_points_);  
                } else if (updata_type == LocalMapUpdataType::TIME_UPDATA) {
                    local_map_->UpdateLocalMapForTime(iter->first, transformed_cloud, res.at(iter->first).nearly_points_);  
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
        typename RegistrationBase<_PointType>::Ptr registration_ptr_;
        typename PointCloudLocalMapBase<_PointType>::Ptr local_map_;  
        Eigen::Isometry3d prev_pose_;  // 上一帧的位姿
        Eigen::Isometry3d curr_pose_;   // 上一帧的位姿
        Eigen::Isometry3d predict_trans_;  // 预测位姿
        Eigen::Isometry3d motion_increment_;  // 运动增量 
        Eigen::Isometry3d last_keyframe_pose_;  // 上一个关键帧的位姿
        double last_keyframe_time_;  // 上一个关键帧的位姿
        LocalMapUpdataType updata_type_; 
}; // class LidarTracker
} // namespace 
}
