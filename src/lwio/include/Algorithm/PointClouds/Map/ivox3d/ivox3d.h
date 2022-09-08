//
// Created by xiang on 2021/9/16.
//

#ifndef FASTER_LIO_IVOX3D_H
#define FASTER_LIO_IVOX3D_H

#include <glog/logging.h>
#include <execution>
#include <list>
#include <thread>
#include <atomic>

#include "eigen_types.h"
#include "ivox3d_node.hpp"

namespace Slam3D {

enum class IVoxNodeType {
    DEFAULT,  // linear ivox
    PHC,      // phc ivox
};

/// traits for NodeType
template <IVoxNodeType node_type, typename PointT, int dim>
struct IVoxNodeTypeTraits {};
// 特化  
template <typename PointT, int dim>
struct IVoxNodeTypeTraits<IVoxNodeType::DEFAULT, PointT, dim> {
    using NodeType = IVoxNode<PointT, dim>;
};

template <typename PointType = pcl::PointXYZ, int dim = 3, 
    IVoxNodeType node_type = IVoxNodeType::DEFAULT>
class IVox {
   public:
    using KeyType = Eigen::Matrix<int, dim, 1>;
    using PtType = Eigen::Matrix<float, dim, 1>;
    using NodeType = typename IVoxNodeTypeTraits<node_type, PointType, dim>::NodeType;
    using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
    using DistPoint = typename NodeType::DistPoint;
    using PointCloud = typename pcl::PointCloud<PointType>; 
    using PointCloudPtr = typename pcl::PointCloud<PointType>::Ptr; 
    enum class NearbyType {
        CENTER,  // center only
        NEARBY6,
        NEARBY18,
        NEARBY26,
    };

    struct Options {
        double resolution_ = 0.2;                        // ivox resolution
        double inv_resolution_ = 10.0;                   // inverse resolution
        NearbyType nearby_type_ = NearbyType::NEARBY6;  // nearby range
        std::size_t capacity_ = 1000000;                // capacity
    };

    /**
     * constructor
     * @param options  ivox options
     */
    explicit IVox(Options options) : options_(options) {
        options_.inv_resolution_ = 1.0 / options_.resolution_;
        GenerateNearbyGrids();
    }

    /**
     * add points
     * @param points_to_add
     */
    void AddPoints(const PointVector& points_to_add);

    void AddPoints(const PointCloud& points_to_add);

    int DeletePoints(const PointCloud& points_to_add);

    int DeletePoints(const PointVector& points_to_add);

    /// get nn
    bool GetClosestPoint(const PointType& pt, PointType& closest_pt) const;

    /// get nn with condition
    bool GetClosestPoint(const PointType& pt, PointVector& closest_pt, 
        uint16_t max_num = 5, double max_range = 5.0) const;

    /// get nn in cloud
    bool GetClosestPoint(const PointVector& cloud, PointVector& closest_cloud) const;

    void GetLocalMapPoints(PointCloudPtr const& pc) const; 

    uint32_t GetLocalMapPointsNum() const; 

    void SetSearchMode(NearbyType const& mode);

    /// get number of points
    size_t NumPoints() const;

    /// get number of valid grids
    size_t NumValidGrids() const;

    /// get statistics of the points
    std::vector<float> StatGridPoints() const;

   private:
    /// generate the nearby grids according to the given options
    void GenerateNearbyGrids();

    /// position to grid
    KeyType Pos2Grid(const PtType& pt) const;

    Options options_;
    // 以三维坐标为key   保存存储体素的链表的迭代器  
    std::unordered_map<KeyType, typename std::list<std::pair<KeyType, NodeType>>::iterator, 
        hash_vec<dim>>  grids_map_;   // voxel hash map
    // 用链表存储所有的体素  
    std::list<std::pair<KeyType, NodeType>> grids_cache_;  // voxel cache
    std::unordered_map<NearbyType, std::vector<KeyType>> nearby_grids_;   // nearbys
};

template <typename PointType, int dim, IVoxNodeType node_type>
bool IVox<PointType, dim, node_type>::GetClosestPoint(const PointType& pt, PointType& closest_pt) const {
    std::vector<DistPoint> candidates;
    auto key = Pos2Grid(ToEigen<float, dim>(pt));
    std::for_each(nearby_grids_[options_.nearby_type_].begin(), nearby_grids_[options_.nearby_type_].end(), 
        [&key, &candidates, &pt, this](const KeyType& delta) {
            auto dkey = key + delta;
            auto iter = grids_map_.find(dkey);
            if (iter != grids_map_.end()) {
                DistPoint dist_point;
                bool found = iter->second->second.NNPoint(pt, dist_point);
                if (found) {
                    candidates.emplace_back(dist_point);
                }
            }
        }
    );

    if (candidates.empty()) {
        return false;
    }

    auto iter = std::min_element(candidates.begin(), candidates.end());
    closest_pt = iter->Get();
    return true;
}

/**
 * @brief: 找到 pt周围的近邻点 
 * @param pt 目标点
 * @param closest_pt 查找结果
 * @param max_num 最大的数量
 * @param max_range 最远考虑的距离 
 * @return {*}
 */
template <typename PointType, int dim, IVoxNodeType node_type>
bool IVox<PointType, dim, node_type>::GetClosestPoint(const PointType& pt, PointVector& closest_pt, 
        uint16_t max_num, double max_range) const {
    closest_pt.clear(); 
    std::vector<DistPoint> candidates;
    // 周围的每个grid 最多提取 max_num 个点 
    candidates.reserve(max_num * nearby_grids_.at(options_.nearby_type_).size());
    auto key = Pos2Grid(ToEigen<float, dim>(pt));

// #define INNER_TIMER
#ifdef INNER_TIMER
    static std::unordered_map<std::string, std::vector<int64_t>> stats;
    if (stats.empty()) {
        stats["knn"] = std::vector<int64_t>();
        stats["nth"] = std::vector<int64_t>();
    }
#endif
    // 先到当前点的voxel中寻找近邻点
    // 最好就是要在当前voxel中找到全部近邻点，这样速度很快，
    // 这样需要有一个很准的预测位姿，因此我们需要融合IMU、轮速计等传感器才行
    auto iter = grids_map_.find(key);
    int num = 0; 
    if (iter != grids_map_.end()) {   // voxel 越小越好   
        num = iter->second->second.KNNPointByCondition(candidates, pt, max_num, max_range);
    }
    // 如果近邻点不足  则去邻居voxel中搜索
    if (num < max_num) {
        // 遍历全部邻居
        for (const KeyType& delta : nearby_grids_.at(options_.nearby_type_)) {
            auto dkey = key + delta;
            iter = grids_map_.find(dkey);
            // 查找邻居的体素 
            if (iter != grids_map_.end()) {
                #ifdef INNER_TIMER
                    auto t1 = std::chrono::high_resolution_clock::now();
                #endif
                // 每个邻居体素 最多找K个最近点 放入  candidates 
                auto tmp = iter->second->second.KNNPointByCondition(candidates, pt, max_num, max_range);
                #ifdef INNER_TIMER
                    auto t2 = std::chrono::high_resolution_clock::now();
                    auto knn = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
                    stats["knn"].emplace_back(knn);
                #endif
            }
        }
    }

    if (candidates.empty()) {
        return false;
    }

#ifdef INNER_TIMER
    auto t1 = std::chrono::high_resolution_clock::now();
#endif

    if (candidates.size() <= max_num) {
    } else {
        // 不进行严格的排序  保证 在 第max_num - 1的位置处，之前的一定小于该值，之后的一定大于该值 
        std::nth_element(candidates.begin(), candidates.begin() + max_num - 1, candidates.end());
        candidates.resize(max_num);
    }
    // 最小的放在最前面 
    std::nth_element(candidates.begin(), candidates.begin(), candidates.end());

#ifdef INNER_TIMER
    auto t2 = std::chrono::high_resolution_clock::now();
    auto nth = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
    stats["nth"].emplace_back(nth);

    constexpr int STAT_PERIOD = 100000;
    if (!stats["nth"].empty() && stats["nth"].size() % STAT_PERIOD == 0) {
        for (auto& it : stats) {
            const std::string& key = it.first;
            std::vector<int64_t>& stat = it.second;
            int64_t sum_ = std::accumulate(stat.begin(), stat.end(), 0);
            int64_t num_ = stat.size();
            stat.clear();
            std::cout << "inner_" << key << "(ns): sum=" << sum_ << " num=" << num_ << " ave=" << 1.0 * sum_ / num_
                      << " ave*n=" << 1.0 * sum_ / STAT_PERIOD << std::endl;
        }
    }
#endif

    closest_pt.clear();
    for (auto& it : candidates) {
        closest_pt.emplace_back(it.Get());
    }
    return closest_pt.empty() == false;
}

template <typename PointType, int dim, IVoxNodeType node_type>
bool IVox<PointType, dim, node_type>::GetClosestPoint(const PointVector& cloud, PointVector& closest_cloud) const {
    std::vector<size_t> index(cloud.size());
    for (int i = 0; i < cloud.size(); ++i) {
        index[i] = i;
    }
    closest_cloud.resize(cloud.size());

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&cloud, &closest_cloud, this](size_t idx) {
        PointType pt;
        if (GetClosestPoint(cloud[idx], pt)) {
            closest_cloud[idx] = pt;
        } else {
            closest_cloud[idx] = PointType();
        }
    });
    return true;
}

template <typename PointType, int dim, IVoxNodeType node_type>
void IVox<PointType, dim, node_type>::SetSearchMode(NearbyType const& mode) {
    options_.nearby_type_ = mode; 
}

template <typename PointType, int dim, IVoxNodeType node_type>
size_t IVox<PointType, dim, node_type>::NumValidGrids() const {
    return grids_map_.size();
}

template <typename PointType, int dim, IVoxNodeType node_type>
void IVox<PointType, dim, node_type>::GenerateNearbyGrids() {
    nearby_grids_[NearbyType::CENTER].clear();
    nearby_grids_[NearbyType::NEARBY6] = {
        //KeyType(0, 0, 0), KeyType(-1, 0, 0), KeyType(1, 0, 0), KeyType(0, 1, 0),
        KeyType(-1, 0, 0), KeyType(1, 0, 0), KeyType(0, 1, 0),
        KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1)};
    nearby_grids_[NearbyType::NEARBY18] = {
        //KeyType(0, 0, 0), KeyType(-1, 0, 0), KeyType(1, 0, 0),   KeyType(0, 1, 0),
        KeyType(-1, 0, 0), KeyType(1, 0, 0),   KeyType(0, 1, 0),
        KeyType(0, -1, 0), KeyType(0, 0, -1), KeyType(0, 0, 1),   KeyType(1, 1, 0),
        KeyType(-1, 1, 0), KeyType(1, -1, 0), KeyType(-1, -1, 0), KeyType(1, 0, 1),
        KeyType(-1, 0, 1), KeyType(1, 0, -1), KeyType(-1, 0, -1), KeyType(0, 1, 1),
        KeyType(0, -1, 1), KeyType(0, 1, -1), KeyType(0, -1, -1)};
    nearby_grids_[NearbyType::NEARBY26] = {
        //KeyType(0, 0, 0), KeyType(-1, 0, 0),  KeyType(1, 0, 0),   KeyType(0, 1, 0),
        KeyType(-1, 0, 0),  KeyType(1, 0, 0),   KeyType(0, 1, 0),
        KeyType(0, -1, 0),  KeyType(0, 0, -1),  KeyType(0, 0, 1),   KeyType(1, 1, 0),
        KeyType(-1, 1, 0),  KeyType(1, -1, 0),  KeyType(-1, -1, 0), KeyType(1, 0, 1),
        KeyType(-1, 0, 1),  KeyType(1, 0, -1),  KeyType(-1, 0, -1), KeyType(0, 1, 1),
        KeyType(0, -1, 1),  KeyType(0, 1, -1),  KeyType(0, -1, -1), KeyType(1, 1, 1),
        KeyType(-1, 1, 1),  KeyType(1, -1, 1),  KeyType(1, 1, -1),  KeyType(-1, -1, 1),
        KeyType(-1, 1, -1), KeyType(1, -1, -1), KeyType(-1, -1, -1)};
}

/**
 * @brief: 迭代添加点云
 */
template <typename PointType, int dim, IVoxNodeType node_type>
void IVox<PointType, dim, node_type>::AddPoints(const PointVector& points_to_add) {
    // 遍历所有点  添加到体素中   
    std::for_each(std::execution::unseq, points_to_add.begin(), points_to_add.end(), 
        [this](const auto& pt) {
            auto key = Pos2Grid(ToEigen<float, dim>(pt));  // grid地图中的三维坐标
            auto iter = grids_map_.find(key);
            // 创立一个新的体素
            if (iter == grids_map_.end()) {
                PointType center;
                center.getVector3fMap() = key.template cast<float>() * options_.resolution_;
                // 最新的体素插入到最前面    IVoxNode  或   IVoxNodePhc 
                grids_cache_.push_front({key, NodeType(center, options_.resolution_)});
                grids_map_.insert({key, grids_cache_.begin()});

                grids_cache_.front().second.InsertPoint(pt);
                // 限制一个grid的数量 
                if (grids_map_.size() >= options_.capacity_) {
                    grids_map_.erase(grids_cache_.back().first);
                    grids_cache_.pop_back();
                }
            } else {
                iter->second->second.InsertPoint(pt);
                // 将该体素剪贴到链表头  最新使用过的调整到链表最前面
                grids_cache_.splice(grids_cache_.begin(), grids_cache_, iter->second);  
                grids_map_[key] = grids_cache_.begin();
            }
        }
    );
    LOG(INFO) << "grids_map_ size:"<<grids_map_.size(); 
}

/**
 * @brief: 迭代添加点云  针对PCL点云的重载 
 */
template <typename PointType, int dim, IVoxNodeType node_type>
void IVox<PointType, dim, node_type>::AddPoints(const PointCloud& points_to_add) {
    // 遍历所有点  添加到体素中   
    std::for_each(std::execution::unseq, points_to_add.points.begin(), points_to_add.points.end(), 
        [this](const auto& pt) {
            auto key = Pos2Grid(ToEigen<float, dim>(pt));  // grid地图中的三维坐标
            auto iter = grids_map_.find(key);
            // 创立一个新的体素
            if (iter == grids_map_.end()) {
                PointType center;
                center.getVector3fMap() = key.template cast<float>() * options_.resolution_;
                // 最新的体素插入到最前面    IVoxNode  或   IVoxNodePhc 
                grids_cache_.push_front({key, NodeType(center, options_.resolution_)});
                grids_map_.insert({key, grids_cache_.begin()});

                grids_cache_.front().second.InsertPoint(pt);
                // 限制一个grid的数量 
                if (grids_map_.size() >= options_.capacity_) {
                    grids_map_.erase(grids_cache_.back().first);
                    grids_cache_.pop_back();
                }
            } else {
                iter->second->second.InsertPoint(pt);
                // 将该体素剪贴到链表头  最新使用过的调整到链表最前面
                grids_cache_.splice(grids_cache_.begin(), grids_cache_, iter->second);  
                grids_map_[key] = grids_cache_.begin();
            }
        }
    );
    LOG(INFO) << "grids_map_ size:"<<grids_map_.size(); 
}

/**
 * @brief: 删除点云
 * @details 删除点对应的voxel内最早的点
 */
template <typename PointType, int dim, IVoxNodeType node_type>
int IVox<PointType, dim, node_type>::DeletePoints(const PointCloud& points_to_add) {
    std::atomic num{0}; 
    int N = 0; 
    // 遍历所有点  添加到体素中   
    // std::for_each(std::execution::unseq, points_to_add.points.begin(), points_to_add.points.end(), 
    //     [&](const auto& pt) {
        for(auto pt : points_to_add.points) {
            auto key = Pos2Grid(ToEigen<float, dim>(pt));  // grid地图中的三维坐标
            auto iter = grids_map_.find(key);

            if (iter != grids_map_.end()) {
                if (iter->second->second.DeletePoint(pt)) {
                    if (iter->second->second.Empty()) {
                        grids_map_.erase(key);
                        grids_cache_.erase(iter->second);
                    }
                    num++; 
                } 
                // else {
                //     LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!delete error"; 
                // }
            } 
            // else {
            //         LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!delete cant find"; 
            // }
            // N++;
        }
        //LOG(INFO) << "N:" << N; 
    // );
    //LOG(INFO) << "grids_map_ size:"<<grids_map_.size(); 
    return num;  
}

/**
 * @brief: 删除点云 重载 
 * @details 删除点对应的voxel内最早的点
 */
template <typename PointType, int dim, IVoxNodeType node_type>
int IVox<PointType, dim, node_type>::DeletePoints(const PointVector& points_to_add) {
    std::atomic num{0};  
    int N = 0; 
    // 遍历所有点  添加到体素中   
    // std::for_each(std::execution::unseq, points_to_add.begin(), points_to_add.end(), 
    //     [&](const auto& pt) {
        for(auto pt : points_to_add) {
            auto key = Pos2Grid(ToEigen<float, dim>(pt));  // grid地图中的三维坐标
            auto iter = grids_map_.find(key);

            if (iter != grids_map_.end()) {
                if (iter->second->second.DeletePoint(pt)) {
                    if (iter->second->second.Empty()) {
                        grids_map_.erase(key);
                        grids_cache_.erase(iter->second);
                    }
                    num++;
                }
                // else {
                //     LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!delete error"; 
                // }
            }  
            // else {
            //         LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!delete cant find"; 
            // }
            // N++;
        }
        //LOG(INFO) << "N:" << N; 
    // );
    //LOG(INFO) << "grids_map_ size:"<<grids_map_.size(); 
    return num; 
}

/**
 * @brief: 获取local map点云
 */
template <typename PointType, int dim, IVoxNodeType node_type>
void IVox<PointType, dim, node_type>::GetLocalMapPoints(PointCloudPtr const& pc) const {
    pc->clear(); 
    //pc->reserve(grids_cache_.size() * 1000); 
    for (auto it = grids_cache_.begin(); it != grids_cache_.end(); ++it) {
        for (std::size_t i = 0; i < it->second.Size(); i++) {
            pc->points.push_back(it->second.GetPoint(i));
        }
    }
}

/**
 * @brief: 获取local map点云
 */
template <typename PointType, int dim, IVoxNodeType node_type>
uint32_t IVox<PointType, dim, node_type>::GetLocalMapPointsNum() const {
    uint32_t num = 0; 
    //pc->reserve(grids_cache_.size() * 1000); 
    for (auto it = grids_cache_.begin(); it != grids_cache_.end(); ++it) {
        num += it->second.Size(); 
    }
    return num;  
}


/**
 * @brief: 获取点在体素地图中的坐标     体素的中心在原点    所以要四舍五入   
 */
template <typename PointType, int dim, IVoxNodeType node_type>
Eigen::Matrix<int, dim, 1> IVox<PointType, dim, node_type>::Pos2Grid(const IVox::PtType& pt) const {
    // return (pt * options_.inv_resolution_).array().round().template cast<int>();  
    return (pt * options_.inv_resolution_).array().floor().template cast<int>();  
}

template <typename PointType, int dim, IVoxNodeType node_type>
std::vector<float> IVox<PointType, dim, node_type>::StatGridPoints() const {
    int num = grids_cache_.size(), valid_num = 0, max = 0, min = 100000000;
    int sum = 0, sum_square = 0;
    for (auto& it : grids_cache_) {
        int s = it.second.Size();
        valid_num += s > 0;
        max = s > max ? s : max;
        min = s < min ? s : min;
        sum += s;
        sum_square += s * s;
    }
    float ave = float(sum) / num;
    float stddev = num > 1 ? sqrt((float(sum_square) - num * ave * ave) / (num - 1)) : 0;
    return std::vector<float>{valid_num, ave, max, min, stddev};
}

}  // namespace Slam3D

#endif
