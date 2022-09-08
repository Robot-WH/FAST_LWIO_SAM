#pragma once

#include "Common/utility.hpp"
#include "LocalMap.hpp"
#include "Algorithm/PointClouds/Map/ivox3d/ivox3d.h"  

namespace Slam3D {
#define IVOX_DEBUG 0
/**
 * @brief:  基于ivox管理的local map  
 * @param _PointType local map中每个地图点的类型  
 */    
template<typename _PointType>
class iVoxLocalMap : public PointCloudLocalMapBase<_PointType> {
protected:
    using Base = PointCloudLocalMapBase<_PointType>; 
    using PointCloudConstPtr = typename pcl::PointCloud<_PointType>::ConstPtr;
    using PointVector = typename Base::PointVector; 
    using KeyType = Eigen::Matrix<int, 3, 1>;  

    struct DownSamplingGrid {
        Eigen::Vector3f center_; // grid 中心坐标
        _PointType point_;    // grid 保存的距离center最近的点
        double dis_ = -1;   // 该最近点到center的距离 
        bool add_new_point_ = false;   // 是否新加了点   
    };
    
public:
    using IvoxNearbyType = typename IVox<_PointType>::NearbyType;
    struct  Option {
        double downsampling_size_;    // 对local map的降采样  
        typename IVox<_PointType>::Options ivox_option_;  
    };

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    iVoxLocalMap(Option option) : option_(option) {
        LOG(INFO) << "create iVoxLocalMap"; 
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief: 由于发生了充足的运动，因此添加一帧数据到Local map   
     * @param name 添加的点云名称
     */            
    void UpdateLocalMapForMotion(std::string const& name, 
            typename pcl::PointCloud<_PointType>::ConstPtr const& frame,
            std::vector<PointVector> const& nearest_points) override {
        updateLocalMap(name, frame, nearest_points); 
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief: 由于长时间没有更新地图，因此添加一帧数据到Local map   
     * @details 此时直接将滑动窗口最末尾的数据移除
     */            
    void UpdateLocalMapForTime(std::string const& name, 
            typename pcl::PointCloud<_PointType>::ConstPtr const& frame,
            std::vector<PointVector> const& nearest_points) override {
        updateLocalMap(name, frame, nearest_points); 
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    virtual bool GetNearlyNeighbor(std::string const& name, _PointType const& point, 
            uint16_t const& num, double const& max_range, PointVector& res) const override {
        if (ivox_container_.find(name) == ivox_container_.end()) return false;   
        // ivox 中 num 表示最多提取的点的数量，而我们这里标识的是最少提取点的数量   
        ivox_container_.at(name).GetClosestPoint(point, res, num, max_range);     
        return res.size() == num;   // 是否搜索到了num个点 
    }

private:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void setSearchNearlyNeighborMode(std::string const& name, 
            typename IVox<_PointType>::NearbyType mode) {
        ivox_container_.at(name).SetSearchMode(mode); 
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief: 添加一帧点云来更新local map 
     * @details: 先进行迭代降采样操作  
     * @param frame 当前需要添加的点云数据(已经和local map 进行了对齐)
     * @param nearest_points frame与localmap匹配的近邻数据 
     * @return {*}
     */            
    void updateLocalMap(std::string const& name,
            typename pcl::PointCloud<_PointType>::ConstPtr const& frame,
            std::vector<PointVector> const& nearest_points) {

        if (ivox_container_.find(name) == ivox_container_.end()) {
            Base::local_map_container_[name].reset(new pcl::PointCloud<_PointType>()); 
            ivox_container_.insert(std::make_pair(name, IVox<_PointType>(option_.ivox_option_)));
        }
        PointVector points_to_add;
        PointVector points_to_delete;  

        int cur_pts = frame->size();
        points_to_add.reserve(cur_pts);
        points_to_delete.reserve(cur_pts);

        // std::vector<size_t> index(cur_pts);
        // for (size_t i = 0; i < cur_pts; ++i) {
        //     index[i] = i;
        // }
        // 降采样 voxel hash map
        std::unordered_map<KeyType, DownSamplingGrid, hash_vec<3>>  downsampling_grid; 

        auto ori_mode = option_.ivox_option_.nearby_type_; 
        float update_point_thresh = 0.2 * option_.downsampling_size_; 
        // 先设定搜索模式    仅仅搜索当前体素 ，注意 要求 搜索体素的size是 滤波体素的整数倍
        //setSearchNearlyNeighborMode(name, IVox<_PointType>::NearbyType::NEARBY26); 
        //std::for_each(std::execution::unseq, index.begin(), index.end(), [&](const size_t &i) {
        for (int i = 0; i < cur_pts; i++) {
            _PointType const& point_world = frame->points[i];
            KeyType grid_key =  
                    (point_world.getVector3fMap() / option_.downsampling_size_).array().floor().template cast<int>();  
            
            if (downsampling_grid.find(grid_key) == downsampling_grid.end()) {
                // LOG(INFO) << "grid_key no have";
                // 落在该体素上的第一个点  
                downsampling_grid[grid_key].center_ = 
                    ((point_world.getVector3fMap() / option_.downsampling_size_).array().floor() + 0.5) * option_.downsampling_size_;
                float curr_point_dist = 
                    Utility::CalcDist(point_world.getVector3fMap(), downsampling_grid[grid_key].center_);
                // 搜索出point_world周围可能在同一个滤波体素的点
                // PointVector points_near;   
                // 搜索出可能和point_world 处与同一个voxel的点
                // 1.74 即 sqrt(3)     32是 最多邻居个数  
                //GetNearlyNeighbor(name, point_world, 32, 1.74 * option_.downsampling_size_, points_near);
                if (!nearest_points.empty() && !nearest_points[i].empty()) {
                //if (!points_near.empty()) {
                    PointVector points_near = nearest_points[i];   
                    // 有localmap匹配数据   接下来需要判断那些localmap要保留,以及当前点是否要添加 
                    // 一个滤波 voxel 内 只保留一个距离中心最近的点  
                    double min_dist = curr_point_dist; 
                    int min_dist_point_index = -1; 

                    for (int k = 0; k < points_near.size(); k++) {
                        // KeyType near_point_key =  
                        //     (points_near[k].getVector3fMap() / option_.downsampling_size_).array().floor().template cast<int>();  
                        // if (near_point_key != grid_key) {
                        //     continue; 
                        // }
                        Eigen::Vector3f point_2_center_vec = 
                            points_near[k].getVector3fMap() - downsampling_grid[grid_key].center_;

                        if (fabs(point_2_center_vec.x()) > 0.5 * option_.downsampling_size_ + 1e-4||
                            fabs(point_2_center_vec.y()) > 0.5 * option_.downsampling_size_ + 1e-4||
                            fabs(point_2_center_vec.z()) > 0.5 * option_.downsampling_size_ + 1e-4) {
                                continue;    // 如果该邻近点不在该voxel 上  那么不需要管
                        }

                        double point_2_center_dis = point_2_center_vec.norm();
                        if (point_2_center_dis < min_dist - update_point_thresh) {
                            if (min_dist_point_index != -1) {
                                // != -1 说明之前的最近点为原有的localmap点,那么该点不能保留 
                                points_to_delete.emplace_back(points_near[min_dist_point_index]); 
                            }
                            min_dist_point_index = k;
                            min_dist = point_2_center_dis;
                        } else {
                            points_to_delete.emplace_back(points_near[k]); 
                        }
                    }
                    // == -1 说明当前点距离其voxel 最近,应该被添加
                    if (min_dist_point_index == -1) {
                        downsampling_grid[grid_key].point_ = point_world;
                        downsampling_grid[grid_key].add_new_point_ = true;  
                    } else {
                        // 记录原最近点, 以便后续进行删除的需求 
                        downsampling_grid[grid_key].point_ = points_near[min_dist_point_index];
                    }
                    downsampling_grid[grid_key].dis_ = min_dist;
                } else {
                    // 没有匹配数据是第一帧   因此是该voxel内的第一个点  
                    downsampling_grid[grid_key].point_ = point_world;
                    downsampling_grid[grid_key].dis_ = curr_point_dist;
                    downsampling_grid[grid_key].add_new_point_ = true;   // 标记为添加  
                }
            } else {
                // 即此前已经有当前帧的点落在同一个voxel 中
                float curr_point_dist = 
                    Utility::CalcDist(point_world.getVector3fMap(), downsampling_grid[grid_key].center_);
                // 当前点 距离更近,那么应该添加当前点  
                if (curr_point_dist < downsampling_grid[grid_key].dis_ - update_point_thresh) {
                    // 如果过去最近点为原Localmap点, 那么应该删除  
                    if (!downsampling_grid[grid_key].add_new_point_) {
                        points_to_delete.emplace_back(downsampling_grid[grid_key].point_); 
                        downsampling_grid[grid_key].add_new_point_ = true;  
                    }
                    downsampling_grid[grid_key].dis_ = curr_point_dist;
                    downsampling_grid[grid_key].point_ = point_world;
                }
            }
        }
        // });
        for (auto& it : downsampling_grid) {
            if (it.second.add_new_point_) {
                points_to_add.emplace_back(it.second.point_);
            }
        }

        ivox_container_.at(name).DeletePoints(points_to_delete);
        ivox_container_.at(name).AddPoints(points_to_add);

        TicToc tt;
        // 获取localmap中点云  
        ivox_container_.at(name).GetLocalMapPoints(Base::local_map_container_[name]);
        tt.toc("update local map points ");

        #if IVOX_DEBUG
            LOG(INFO) << "check ivox ............................"; 
            int check_num = 0; 
            // 检查
            std::unordered_map<KeyType, DownSamplingGrid, hash_vec<3>>  downsampling_grid_2; 
            for (int i = 0; i < Base::local_map_container_[name]->size(); i++) {
                _PointType const& point_world = Base::local_map_container_[name]->points[i];
                KeyType grid_key =  
                        (point_world.getVector3fMap() / option_.downsampling_size_).array().floor().template cast<int>();  
                if (downsampling_grid_2.find(grid_key) == downsampling_grid_2.end()) {
                    downsampling_grid_2[grid_key].point_ = point_world;
                } else {
                    check_num++; 
                    PointVector nearest_p;
                    LOG(INFO) << "grid_key :" << grid_key.transpose();
                    LOG(INFO) << "center :" << downsampling_grid[grid_key].center_.transpose();
                    Eigen::Vector3f point_2_center_vec = 
                            downsampling_grid_2[grid_key].point_.getVector3fMap() - downsampling_grid[grid_key].center_;
                    LOG(INFO) << "repeat, ori :" << downsampling_grid_2[grid_key].point_.getVector3fMap().transpose()
                    <<",dis to center: " << point_2_center_vec.transpose();
                    if (fabs(point_2_center_vec.x()) > 0.5 * option_.downsampling_size_||
                            fabs(point_2_center_vec.y()) > 0.5 * option_.downsampling_size_||
                            fabs(point_2_center_vec.z()) > 0.5 * option_.downsampling_size_) {
                        LOG(INFO) << "out the voxel";   
                    }
                    GetNearlyNeighbor(name, downsampling_grid_2[grid_key].point_, 32, 1.74 * option_.downsampling_size_, nearest_p);
                    for (auto const& it : nearest_p) {
                        LOG(INFO) << "nearest_p it:" << it.getVector3fMap().transpose();
                    }

                    point_2_center_vec = 
                            point_world.getVector3fMap() - downsampling_grid[grid_key].center_;
                    if (fabs(point_2_center_vec.x()) > 0.5 * option_.downsampling_size_||
                            fabs(point_2_center_vec.y()) > 0.5 * option_.downsampling_size_||
                            fabs(point_2_center_vec.z()) > 0.5 * option_.downsampling_size_) {
                        LOG(INFO) << "out the voxel";   
                    }
                    LOG(INFO) << "repeat, new :" << point_world.getVector3fMap().transpose()
                    <<",dis to center: " << point_2_center_vec.transpose(); 
                    GetNearlyNeighbor(name, point_world, 5, 1, nearest_p);
                    for (auto const& it : nearest_p) {
                        LOG(INFO) << "nearest_p it:" << it.getVector3fMap().transpose();
                    }

                }
            }
            if (check_num > 0) {
                LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! check_num: " << check_num; 
            }
        #endif
        // 恢复搜索模式
        //setSearchNearlyNeighborMode(name, ori_mode); 
    }

    Option option_; 
    std::unordered_map<std::string, IVox<_PointType>> ivox_container_;  
}; // class iVoxLocalMap
}

