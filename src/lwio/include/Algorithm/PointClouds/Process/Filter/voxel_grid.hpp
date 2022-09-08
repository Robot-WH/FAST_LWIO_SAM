
#pragma once 

#include "filter_base.hpp"
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

namespace Slam3D {
namespace Algorithm {
namespace Filter {

    template<class _PointT>
    using  VoxelGridPtr = std::unique_ptr<pcl::VoxelGrid<_PointT>>;  

    struct VoxelGridOption {
        float resolution_ = 0;  
    };
    struct ApproximateVoxelGridOption {
    };

    /**
     * @brief: 构建降采样滤波器 voxel  grid 
     * @details 用voxel 中的所有点的均值代替其他点    
     * @param resolution voxel 的分辨率  
     */
    template<class _PointT>
    VoxelGridPtr<_PointT> CreateVoxelGridFilter(VoxelGridOption option) {
        LOG(INFO) << "create VoxelGridFilter!";
        LOG(INFO) << "VoxelGridFilter resolution: "<<option.resolution_;
        VoxelGridPtr<_PointT> voxelgrid(new pcl::VoxelGrid<_PointT>());
        voxelgrid->setLeafSize(option.resolution_, option.resolution_, option.resolution_);
        return std::move(voxelgrid);   
    }

    template<typename _PointType>
    class VoxelGridFilter : public FilterBase<_PointType> {
        public:
            struct Option {
                std::string mode_;
                VoxelGridOption voxel_grid_option_;
                ApproximateVoxelGridOption approximate_voxel_grid_option_;
            }; 
            VoxelGridFilter() {}  

            VoxelGridFilter(Option option) {
                Reset(option); 
            }
            /**
             * @brief: 对降采样的模式进行设置
             * @param mode 滤波器的模式 
             * @return {*}
             */            
            void Reset(Option option) {
                if (option.mode_ == "VoxelGrid") {
                    FilterBase<_PointType>::SetFilter(CreateVoxelGridFilter<_PointType>(option.voxel_grid_option_)); 
                } else if (option.mode_ == "ApproximateVoxelGrid") {
                }
            }
    };
} // namespace Filter
} // namespace Algorithm
}

