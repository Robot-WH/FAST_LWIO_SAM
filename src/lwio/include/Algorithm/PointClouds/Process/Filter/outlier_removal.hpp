/*
 * @Copyright(C): Your Company
 * @FileName: 文件名
 * @Author: 作者
 * @Version: 版本
 * @Date: 2022-02-28 12:46:49
 * @Description: 
 * @Others: 
 */

#pragma once 
#include "filter_base.hpp"
#include "factory/processing/pointcloud/filter/filter_factory.hpp"

namespace Slam3D {
namespace Algorithm {
namespace Filter { 

    struct RadiusOutlierOption {
        float radius_;  // 考虑的范围
        uint16_t min_neighbors_;  // 范围内最小邻居数  
    };
    struct StatisticalOutlierOption {
        uint16_t mean_k_;
        uint8_t k_;
    };

    template<class PointT>
    using RadiusOutlierRemovalPtr = std::unique_ptr<pcl::RadiusOutlierRemoval<PointT>>;  
    template<class PointT>
    using  StatisticalOutlierRemovalPtr = std::unique_ptr<pcl::StatisticalOutlierRemoval<PointT>>;  

    /**
     * @brief: 构建几何离群点滤波  
     * @details 
     * @param radius 考虑的半径 
     * @param min_neighbors 半径内最少的点的数量 
     */
    template<class PointT>
    RadiusOutlierRemovalPtr<PointT> CreateRadiusOutlierRemoval(RadiusOutlierOption option)  {
        RadiusOutlierRemovalPtr<PointT> rad(new pcl::RadiusOutlierRemoval<PointT>());
        rad->setRadiusSearch(option.radius_);                                         
        rad->setMinNeighborsInRadius(option.min_neighbors_);  
        return std::move(rad);   
    }

    /**
     * @brief: 构建统计离群点滤波  
     * @details  先遍历所有点，统计每个点距离最近的mean_k个邻居的距离，计算该距离的正态分布系数，
     *                      再遍历所有点，如果 每个点距离最近的mean_k个邻居的平均距离 在 μ ±k*σ  之外,
     *                      则滤除   
     * @param mean_k 考虑的邻居个数 
     * @param k 需要滤除的点的平均距离大于标准差的倍数 μ ±k*σ  之外的点  滤除
     */
    template<class PointT>
    StatisticalOutlierRemovalPtr<PointT> CreateStatisticalOutlierRemoval(StatisticalOutlierOption option) {
        StatisticalOutlierRemovalPtr<PointT> sor(new pcl::StatisticalOutlierRemoval<PointT>());
        sor->setMeanK (option.mean_k_);    
        sor->setStddevMulThresh (option.k_);
        return std::move(sor);  
    }

    template<typename _PointType>
    class OutlierRemovalFilter : public FilterBase<_PointType> {
        public:
            struct Option {
                std::string mode_;  
                RadiusOutlierOption radiusOutlier_option_; 
                StatisticalOutlierOption statisticalOutlier_option_; 
            }; 
            OutlierRemovalFilter() {}
            OutlierRemovalFilter(Option option) {
                Reset(option);
            }
            void Reset(Option option) {
                if (option.mode_ == "radiusOutlier") {
                    FilterBase<_PointType>::SetFilter(
                        CreateRadiusOutlierRemoval<_PointType>(option.radiusOutlier_option_)); 
                } else if (option.mode_ == "statisticalOutlier") {
                    FilterBase<_PointType>::SetFilter(
                        CreateStatisticalOutlierRemoval<_PointType>(option.statisticalOutlier_option_)); 
                }
            }
    };
} // namepace Filter
} // namespace Algorithm
} // namespace Slam3D 
