#pragma once 

#include "FeatureMatchBase.hpp"

namespace Slam3D {
namespace Algorithm {

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename _PointT>
class EdgeFeatureMatch : public FeatureMatch<_PointT> {
    private:
        using base = FeatureMatch<_PointT>;  
        using PointVector = typename base::PointVector;
    public:
        // 特征匹配后的信息    
        struct EdgeCostFactorInfo {
            bool is_valid_ = false;  
            Eigen::Vector3d norm_ = {0, 0, 0};
            double residuals_ = 0;
            double s_ = 0;     // 权重
            std::vector<Eigen::Vector3d> points_edge_; 
            PointVector matched_points_;    // 全部匹配的点 
        }; 

        EdgeFeatureMatch(std::string const& label) : base(label) {}

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        bool Match(_PointT const& point, EdgeCostFactorInfo &res) {
            if (!base::SearchKNN(point, 5, base::search_thresh_, res.matched_points_)) return false;   
            // 进行PCA
            std::vector<Eigen::Vector3d> nearCorners;
            Eigen::Vector3d center(0, 0, 0);

            for (int j = 0; j < 5; j++) {
                Eigen::Vector3d tmp(res.matched_points_[j].x, res.matched_points_[j].y, res.matched_points_[j].z);
                center = center + tmp;
                nearCorners.push_back(std::move(tmp));
            }
            center = center / 5.0;
            // 计算协方差矩阵
            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
            for (int j = 0; j < 5; j++) {
                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
            }
            // 特征分解 
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

            Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);
            // 计算线性度  
            // 要足够像一条线   
            if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1]) { 
                Eigen::Vector3d point_on_line = center;
                Eigen::Vector3d point_a, point_b;
                point_a = 0.1 * unit_direction + point_on_line;
                point_b = -0.1 * unit_direction + point_on_line;
                Eigen::Vector3d curr_point(point.x, point.y, point.z);
                // 计算匹配的直线方程  
                Eigen::Vector3d nu = (curr_point - point_a).cross(curr_point - point_b);   // 平行四边形的面积 
                Eigen::Vector3d de = point_a - point_b;  
                double de_norm = de.norm();
                res.residuals_ = nu.norm() / de_norm;    // 点到线的距离    
                res.norm_ = de.cross(nu).normalized();  // 残差的法向量 
                res.points_edge_[0] = point_a;
                res.points_edge_[1] = point_b;  
                res.is_valid_ = true;  
                return true;
            }                                                                           
            return false;          
        }
};
}
}
