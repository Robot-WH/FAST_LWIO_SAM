#pragma once 
#include <pcl/point_cloud.h>
#include <pcl/common/centroid.h>
#include <algorithm>
#include <cmath>
#include <list>
#include <vector>

namespace Slam3D {

// squared distance of two pcl points
template <typename PointT>
inline double distance2(const PointT& pt1, const PointT& pt2) {
    Eigen::Vector3f d = pt1.getVector3fMap() - pt2.getVector3fMap();
    return d.squaredNorm();
}

// convert from pcl point to eigen
template <typename T, int dim, typename PointType>
inline Eigen::Matrix<T, dim, 1> ToEigen(const PointType& pt) {
    return Eigen::Matrix<T, dim, 1>(pt.x, pt.y, pt.z);
}

template <>
inline Eigen::Matrix<float, 3, 1> ToEigen<float, 3, pcl::PointXYZ>(const pcl::PointXYZ& pt) {
    return pt.getVector3fMap();
}

template <>
inline Eigen::Matrix<float, 3, 1> ToEigen<float, 3, pcl::PointXYZI>(const pcl::PointXYZI& pt) {
    return pt.getVector3fMap();
}

template <>
inline Eigen::Matrix<float, 3, 1> ToEigen<float, 3, pcl::PointXYZINormal>(const pcl::PointXYZINormal& pt) {
    return pt.getVector3fMap();
}

/**
 * @brief: 一个体素
 */
template <typename PointT, int dim = 3>
class IVoxNode {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        struct DistPoint;

        IVoxNode() = default;
        IVoxNode(const PointT& center, const float& side_length) {}  /// same with phc

        void InsertPoint(const PointT& pt);

        inline bool DeleteEarliestPoint(); 

        inline bool DeletePoint(PointT const& point);

        inline bool Empty() const;

        inline std::size_t Size() const;

        inline PointT GetPoint(const std::size_t idx) const;

        int KNNPointByCondition(std::vector<DistPoint>& dis_points, const PointT& point, const int& K,
                                const double& max_range);

    private:
        std::vector<PointT> points_;   // 存储体素内全部的点
        //pcl::PointCloud<PointT> point_cloud_; 
};

/**
 * @brief: 体素中某个点以及与目标点的距离的信息
 * @details: 
 * @return {*}
 */
template <typename PointT, int dim>
struct IVoxNode<PointT, dim>::DistPoint {
    double dist = 0;
    IVoxNode* node = nullptr;    // 所属的体素 node 指针
    int idx = 0;

    DistPoint() = default;
    DistPoint(const double d, IVoxNode* n, const int i) : dist(d), node(n), idx(i) {}

    PointT Get() { return node->GetPoint(idx); }
    inline bool operator()(const DistPoint& p1, const DistPoint& p2) { return p1.dist < p2.dist; }
    inline bool operator<(const DistPoint& rhs) { return dist < rhs.dist; }
};

template <typename PointT, int dim>
void IVoxNode<PointT, dim>::InsertPoint(const PointT& pt) {
    points_.template emplace_back(pt);
}

/**
 * @brief: 删除最早的一个点
 * @return 点容器是否为空
 */
template <typename PointT, int dim>
bool IVoxNode<PointT, dim>::DeleteEarliestPoint() {
    //points_.template pop_front();
    points_.template erase(points_.begin());
    return points_.empty();
}

/**
 * @brief: 删除一个点 (与point距离足够接近的点)
 * @return 点容器是否为空
 */
template <typename PointT, int dim>
bool IVoxNode<PointT, dim>::DeletePoint(PointT const& point) {
    for (auto iter = points_.begin(); iter != points_.end(); ++iter) {
        Eigen::Vector3f diff = point.getVector3fMap() - iter->getVector3fMap();
        // 每个维度的差异小于1mm即可
        if (fabs(diff.x()) < 0.001 &&fabs(diff.y()) < 0.001 && fabs(diff.z()) < 0.001) {
                //points_.template pop_front();
                points_.template erase(iter);
                return true; 
        }
    }
    return false;
}

template <typename PointT, int dim>
bool IVoxNode<PointT, dim>::Empty() const {
    return points_.empty();
}

template <typename PointT, int dim>
std::size_t IVoxNode<PointT, dim>::Size() const {
    return points_.size();
}

template <typename PointT, int dim>
PointT IVoxNode<PointT, dim>::GetPoint(const std::size_t idx) const {
    return points_[idx];
}

/**
 * @brief: 在voxel 中 找出 距离 point 小于 max_range 的点
 * @details: 最多找K个
 * @param dis_points 存放找到的点的信息
 * @param point 目标点
 * @param K 最多找的点的个数
 * @param max_range 近邻点的最大距离 
 * @return {*}
 */
template <typename PointT, int dim>
int IVoxNode<PointT, dim>::KNNPointByCondition(std::vector<DistPoint>& dis_points, 
        const PointT& point, const int& K,  const double& max_range) {
    std::size_t old_size = dis_points.size();
// #define INNER_TIMER
#ifdef INNER_TIMER
    static std::unordered_map<std::string, std::vector<int64_t>> stats;
    if (stats.empty()) {
        stats["dis"] = std::vector<int64_t>();
        stats["put"] = std::vector<int64_t>();
        stats["nth"] = std::vector<int64_t>();
    }
#endif

    // 遍历 体素中 全部的点  找到
    for (const auto& pt : points_) {
#ifdef INNER_TIMER
        auto t0 = std::chrono::high_resolution_clock::now();
#endif
        double d = distance2(pt, point);
#ifdef INNER_TIMER
        auto t1 = std::chrono::high_resolution_clock::now();
#endif
        if (d < max_range * max_range) {
            // vector data()  - 该函数返回一个指向数组中第一个元素的指针
            // &pt - points_.data() 即 pt在数组中的序号
            dis_points.template emplace_back(DistPoint(d, this, &pt - points_.data()));
        }
#ifdef INNER_TIMER
        auto t2 = std::chrono::high_resolution_clock::now();

        auto dis = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        stats["dis"].emplace_back(dis);
        auto put = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        stats["put"].emplace_back(put);
#endif
        //index++; 
    }
#ifdef INNER_TIMER
    auto t1 = std::chrono::high_resolution_clock::now();
#endif
    // sort by distance
    if (old_size + K >= dis_points.size()) {   // 点数少于K个那么全部都要
    } else {
        // 每一个体素 最多选取 K 个点，因此，这里找出前K个距离最小的点 
        std::nth_element(dis_points.begin() + old_size, dis_points.begin() + old_size + K - 1, dis_points.end());
        dis_points.resize(old_size + K);
    }

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

    return dis_points.size();
}

}  // namespace Slam3D
