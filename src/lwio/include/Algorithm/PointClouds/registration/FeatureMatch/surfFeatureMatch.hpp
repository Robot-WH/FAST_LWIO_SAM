
#pragma once 

#include "FeatureMatchBase.hpp"
namespace Slam3D {
namespace Algorithm {

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename _PointT>
class SurfFeatureMatch : public FeatureMatch<_PointT> {
    private:
        using Base = FeatureMatch<_PointT>;  
        using PointVector = typename Base::PointVector;
    public:
        // 特征匹配后的信息    
        struct SurfCostFactorInfo {
            bool is_valid_ = false;  
            Eigen::Vector3d norm_ = {0, 0, 0};
            double D_ = 0;  
            double residuals_ = 0;
            double s_ = 0;     // 权重
            PointVector matched_points_;  
        }; 
        // label 即匹配器 匹配点云的标识名
        SurfFeatureMatch(std::string const& label) : Base(label) {}

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        bool Match(_PointT const& point, SurfCostFactorInfo &res) {
            if (!Base::SearchKNN(point, 5, Base::search_thresh_, res.matched_points_)) 
                return false;   
            Eigen::Matrix<double, 5, 3> matA0;
            Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();
            // 首先拟合平面   
            for (int j = 0; j < 5; j++) {
                matA0(j, 0) = res.matched_points_[j].x;
                matA0(j, 1) = res.matched_points_[j].y;
                matA0(j, 2) = res.matched_points_[j].z;
            }
            // find the norm of plane
            Eigen::Vector3d norm = matA0.colPivHouseholderQr().solve(matB0);    // 法向量 n ( 未归一化)
            double D = 1 / norm.norm();   // D  
            norm.normalize(); // 归一化法向量 
            // 判断该平面质量   
            bool planeValid = true;
            for (int j = 0; j < 5; j++) {
                // if OX * n > 0.2, then plane is not fit well
                if (fabs(norm(0) * res.matched_points_[j].x + 
                        norm(1) * res.matched_points_[j].y + 
                        norm(2) * res.matched_points_[j].z + D) > 0.2) {
                    planeValid = false;
                    break;
                }
            }
            // 质量好  则匹配成功   计算出匹配的残差  
            if (planeValid) {
                Eigen::Vector3d curr_point(point.x, point.y, point.z);
                float distance = norm.dot(curr_point) + D;
                // 残差  
                // res.residuals_ = std::fabs(distance);
                
                // if (distance >= 0) {
                //     res.norm_ = norm;
                //     res.D_ = D;  
                // } else {
                //     res.norm_ = -norm;
                //     res.D_ = -D;  
                // }
                res.residuals_ = distance;
                res.norm_ = norm;
                res.D_ = D;  
                res.is_valid_ = true;  
                return true;
            }
            return false;          
        }
};
}
}
