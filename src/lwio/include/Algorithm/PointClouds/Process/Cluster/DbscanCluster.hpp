
#pragma once 
#include <pcl/point_types.h>

namespace Slam3D {
namespace Algorithm {

template <typename _PointT>
class DBSCANCluster {
    public:
        using Ptr = std::unique_ptr<DBSCANCluster<_PointT>>;
        using PointCloudPtr = typename pcl::PointCloud<_PointT>::Ptr;
        struct Option {
            uint16_t min_density_ = 5;   // 密度 ：一定范围内的点的数量
            uint16_t min_pts_per_cluster_ = 10; // 合法聚类的最小点数量 
            float search_range_coeff_ = 0.01; // 转换系数  ， 搜索范围 = search_range_coeff_ * range 
            float max_search_range = 0.5; // 最大搜索范围
        };

        DBSCANCluster(Option const& option) : option_(option) {}

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 提取聚类
         * @param cloud_in
         * @param cluster_indices 有效聚类的序号
         * @param outlier_indices 无效聚类(很小)的序号 
         * @return {*}
         */        
        void Extract(typename pcl::PointCloud<_PointT>::ConstPtr const& cloud_in, 
                std::vector< std::vector<uint32_t>>& cluster_indices, 
                std::vector< std::vector<uint32_t>>& outlier_indices) {
            cluster_indices.clear();
            outlier_indices.clear();  
            std::vector<int> nn_indices;
            std::vector<float> nn_distances;
            std::vector<bool> is_outliers(cloud_in->points.size(), false);
            std::vector<PointType> types(cloud_in->points.size(), PointType::UN_PROCESSED);
            std::vector<uint32_t> seed_queue;
            seed_queue.reserve(cloud_in->points.size()); 
            // 构造kdtree
            search_tree_.setInputCloud(cloud_in);
            float search_range = 0; 
            
            for (int i = 0; i < cloud_in->points.size(); i++) {
                if (types[i] == PointType::PROCESSED) {
                    continue;
                }
                search_range = cloud_in->points[i].range * option_.search_range_coeff_; 
                if (search_range > option_.max_search_range) {
                    search_range = option_.max_search_range;
                }
                int nn_size = search_tree_.radiusSearch(cloud_in->points[i], 
                    search_range, nn_indices, nn_distances); 
                if (nn_size < option_.min_density_) {
                    // 稀疏点提前进行标记  这样以后遇见该点就不用进行搜索了
                    is_outliers[i] = true;
                    continue;
                }
                // 密度够大  认为是一个聚类 
                // 第一个种子
                seed_queue.push_back(i);
                types[i] = PointType::PROCESSED;
                // 添加该种子周围的邻近点 
                for (int j = 0; j < nn_size; j++) {
                    if (types[nn_indices[j]] == PointType::PROCESSED) continue;   
                    seed_queue.push_back(nn_indices[j]);    
                    types[nn_indices[j]] = PointType::PROCESSED;
                } 
                // 区域生长  
                uint32_t sq_idx = 1;
                while (sq_idx < seed_queue.size()) {
                    int cloud_index = seed_queue[sq_idx];
                    if (is_outliers[cloud_index]) {
                        sq_idx++;
                        continue; // no need to check neighbors.
                    }
                    search_range = cloud_in->points[cloud_index].range * option_.search_range_coeff_; 
                    if (search_range > option_.max_search_range) {
                        search_range = option_.max_search_range;
                    }
                    nn_size = search_tree_.radiusSearch(cloud_in->points[cloud_index], 
                        search_range, nn_indices, nn_distances); 
                    if (nn_size >= option_.min_density_) {
                        for (int j = 0; j < nn_size; j++) {
                            if (types[nn_indices[j]] == PointType::UN_PROCESSED) {
                                seed_queue.push_back(nn_indices[j]);
                                types[nn_indices[j]] = PointType::PROCESSED;
                            }
                        }
                    }
                    sq_idx++;
                }
                if (seed_queue.size() >= option_.min_pts_per_cluster_) {       
                    cluster_indices.push_back(std::move(seed_queue)); 
                } else {    
                    outlier_indices.push_back(std::move(seed_queue)); 
                }
                seed_queue.clear();  
            }
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void setMinClusterSize (int min_cluster_size) { 
            option_.min_pts_per_cluster_ = min_cluster_size; 
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void setCorePointMinPts(int core_point_min_pts) {
            option_.min_density_ = core_point_min_pts;
        }

    protected:
        enum class PointType {
            UN_PROCESSED = 0,
            PROCESSED
        };
        Option option_;  
        PointCloudPtr input_cloud_;
        pcl::KdTreeFLANN<_PointT> search_tree_;
}; // class DBSCANCluster
}
}