
#pragma once 
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/transforms.h>
#include <eigen3/Eigen/Dense>
#include "Map/LocalMap.hpp"

namespace Slam3D {
namespace Algorithm {

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: 匹配器  两种匹配目标   1、与点云匹配    2、与local map进行匹配   
 * @details: 1、与点云匹配，则自己构造kdtree进行搜索  2、和local map进行匹配则使用localmap的搜索函数 
 * @return {*}
 */
template<typename _PointT>
class FeatureMatch {
    public:
        using PointCloudConstPtr = typename pcl::PointCloud<_PointT>::ConstPtr;  
        using LocalMapConstPtr = typename PointCloudLocalMapBase<_PointT>::ConstPtr;  
        using KdtreePtr = typename pcl::KdTreeFLANN<_PointT>::Ptr; 
        using PointVector = std::vector<_PointT, Eigen::aligned_allocator<_PointT>>;
        enum class searchType {
            no_target, 
            localmap, 
            pointcloud
        } search_type_ = searchType::no_target;

        FeatureMatch(std::string const& label) : label_(label), target_points_(nullptr) {
            search_tree_ = KdtreePtr(new pcl::KdTreeFLANN<_PointT>());
        }
        virtual ~FeatureMatch() {}
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 设置匹配目标为点云
         */        
        virtual void SetSearchTarget(PointCloudConstPtr const& target_ptr) {
            search_tree_->setInputCloud(target_ptr);    // 构造kdtree
            target_points_ = target_ptr;
            search_type_ = searchType::pointcloud;
        } 
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 设置匹配目标为local map  
         */        
        virtual void SetSearchTarget(LocalMapConstPtr const& target_ptr) {
            search_local_map_ = target_ptr;  
            search_type_ = searchType::localmap;
        } 
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: KNN搜索  到target中搜索距离point最近的点
         * @param num 需要搜索的数量   
         * @param max_range 最远有效匹配距离  
         * @param res 搜索的到的点的集合   
         * @return 是否搜索到了num 个近邻点 
         */        
        virtual bool SearchKNN(_PointT const& point, uint16_t const& num, 
                double const& max_range, PointVector& res) {
            if (search_type_ == searchType::localmap) {
                // 去local map中搜索近邻点  
                return search_local_map_->GetNearlyNeighbor(label_, point, num, max_range, res);
            } 
            if (search_type_ == searchType::pointcloud) {
                std::vector<int> pointSearchInd;
                std::vector<float> pointSearchSqDis;
                search_tree_->nearestKSearch(point, num, pointSearchInd, pointSearchSqDis); 
                double sq_max_range = max_range * max_range;
                res.clear();
                for (uint16_t i = 0; i < pointSearchInd.size(); i++) {
                    if (pointSearchSqDis[i] > sq_max_range) break; 
                    res.emplace_back(target_points_->points[pointSearchInd[i]]);
                }
                return (res.size() == num);     
            }
            return false; 
        } 

    protected:
        //kd-tree
        KdtreePtr search_tree_;
        LocalMapConstPtr search_local_map_;
        typename pcl::PointCloud<_PointT>::ConstPtr target_points_;
        float search_thresh_ = 1.0;    // 最远的特征点搜索范围
        std::string label_;    // 匹配特征的标识名
}; // class FeatureMatch
} // namespace Algorithm
} // namespace Slam3D 
