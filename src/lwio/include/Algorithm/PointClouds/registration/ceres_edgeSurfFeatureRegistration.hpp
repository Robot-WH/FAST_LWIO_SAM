#pragma once 

#include "registration_base.hpp"
#include "ceres_factor/edge_factor.hpp"
#include "ceres_factor/surf_factor.hpp"
#include "Algorithm/Ceres/Parameterization/PoseSE3Parameterization.hpp"
#include "tic_toc.h"
#include "FeatureMatch/EdgeFeatureMatch.hpp"
#include "FeatureMatch/surfFeatureMatch.hpp"
#include <execution>  // C++ 17 并行算法 
#include <mutex>
#include <atomic>

namespace Slam3D {
namespace Algorithm {

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: 基于ceres求解的线面特征ICP
 */
template<typename _PointType>
class CeresEdgeSurfFeatureRegistration : public RegistrationBase<_PointType> {
    private:
        using Base = RegistrationBase<_PointType>; 
        using RegistrationResult = typename Base::RegistrationResult;  
    public:
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 构造
         * @param edge_label 边特征的标识名
         * @param surf_label 面特征的标识名
         */        
        CeresEdgeSurfFeatureRegistration(std::string const& edge_label, std::string const& surf_label)
        : edge_label_(edge_label), surf_label_(surf_label), optimization_count_(10)
        , surf_point_in_(nullptr), edge_point_in_(nullptr)
        ,edge_match_(edge_label_), surf_match_(surf_label_) {
            points_registration_res_.insert(std::make_pair(edge_label_, typename Base::pointRegistrationResult())); 
            points_registration_res_.insert(std::make_pair(surf_label_, typename Base::pointRegistrationResult())); 
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 传入
         * @details:  匹配逻辑是source与target匹配 
         * @param target_input  点云名称+点云数据 std::pair<std::string, PointCloudPtr> 
         */            
        void SetInputTarget(FeaturePointCloudContainer<_PointType> const& target_input) override {
            // TicToc tt;
            // tt.tic(); 
            if (target_input.find(edge_label_) != target_input.end()) {
                edge_match_.SetSearchTarget(target_input.find(edge_label_)->second);  
            }
            if (target_input.find(surf_label_) != target_input.end()) {
                surf_match_.SetSearchTarget(target_input.find(surf_label_)->second);  
            }
            // tt.toc("kdtree "); 
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 重载 传入 localmap 进行匹配
         * @param target_input  点云名称+点云数据 std::pair<std::string, PointCloudPtr> 
         */            
        void SetInputTarget(typename Base::LocalMapConstPtr const& target_input) override {
            // TicToc tt;
            // tt.tic(); 
            edge_match_.SetSearchTarget(target_input);  
            surf_match_.SetSearchTarget(target_input);  
            // tt.toc("kdtree "); 
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 
         * @details: 
         * @param {FeaturePointCloudContainer<_PointType>} const
         * @return {*}
         */        
        void SetInputSource(FeaturePointCloudContainer<_PointType> const& source_input) override {   
            // 直接使用点云数据 
            if (source_input.find(edge_label_) != source_input.end()) {
                edge_point_in_ = source_input.find(edge_label_)->second; 
            }
            if (source_input.find(surf_label_) != source_input.end()) {
                surf_point_in_ = source_input.find(surf_label_)->second; 
            }
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void SetMaxIteration(uint16_t const& n) override {
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void SetNormIteration(uint16_t const& n) override {
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 求解匹配 
         * @details 
         * @param[out] T 输入匹配预测初值   返回匹配的结果
         * @return {*}
         */            
        void Solve(Eigen::Isometry3d &T) override {
            TicToc tt;  
            tt.tic();  
            //LOG(INFO) << "Solve";
            if (optimization_count_ > 3) 
                optimization_count_--;   
            // 将预测位姿设定为优化前的初始值    
            q_w_curr = Eigen::Quaterniond(T.rotation());
            t_w_curr = T.translation();
            bool save_match_info = false;     // 保存匹配信息 
            // 迭代
            for (int iterCount = 0; iterCount < optimization_count_; iterCount++) {
                ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                ceres::Problem::Options problem_options;
                ceres::Problem problem(problem_options);
                problem.AddParameterBlock(parameters, 7, new PoseSE3Parameterization());
                if (iterCount == optimization_count_ - 1) {
                    save_match_info = true;   // 最后一次迭代需要保存匹配的信息 
                }
                TicToc tt;
                // 进行tbb加速
                // 耗时的重点！！！
                addEdgeCostFactor(problem, loss_function, save_match_info);
                addSurfCostFactor(problem, loss_function, save_match_info);
                tt.toc("addCostFactor ");
                ceres::Solver::Options options;
                options.linear_solver_type = ceres::DENSE_QR;
                options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT; 
                options.max_num_iterations = 4;
                options.minimizer_progress_to_stdout = false;
                options.check_gradients = false;
                options.gradient_check_relative_precision = 1e-4;
                ceres::Solver::Summary summary;
                tt.tic();
                ceres::Solve(options, &problem, &summary);   // 耗时  
                tt.toc("Solve");
            }
            // 判断退化
            T = Eigen::Isometry3d::Identity();
            T.linear() = q_w_curr.toRotationMatrix();
            T.translation() = t_w_curr;
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 获取本次匹配的结果信息  
         * @details: 即每一点的匹配残差以及匹配点 
         * @return {*}
         */        
        RegistrationResult const& GetRegistrationResult() const override {
            // LOG(INFO) << "points_registration_res_[surf_label_].nearly_points_ size: "
            // <<points_registration_res_.at(surf_label_).nearly_points_.size(); 
            return points_registration_res_; 
        }

    protected:
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // 加速前：约51ms  
        // 加速后：约33ms 
        void addSurfCostFactor(ceres::Problem& problem, ceres::LossFunction *loss_function, 
                bool const& save_match_info) {
            if (surf_point_in_ == nullptr || surf_point_in_->empty()) return;  
            if (save_match_info) {
                points_registration_res_[surf_label_].residuals_.clear();  
                points_registration_res_[surf_label_].nearly_points_.clear();  
                points_registration_res_[surf_label_].residuals_.resize(surf_point_in_->points.size()); 
                points_registration_res_[surf_label_].nearly_points_.resize(surf_point_in_->points.size()); 
            }
            std::atomic<uint32_t> surf_num{0};
            //std::cout<<"addSurfCostFactor, points.size(): "<<(int)surf_point_in_->points.size()<<std::endl;
            std::vector<uint32_t> index(surf_point_in_->points.size());
            for (uint32_t i = 0; i < index.size(); ++i) {
                index[i] = i;
            }
            std::vector<ceres::CostFunction*> factor_container(surf_point_in_->points.size(), nullptr);
            //for (int i = 0; i < (int)surf_point_in_->points.size(); i++) {
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), 
                [&](const size_t &i) {
                    _PointType point_temp;
                    pointAssociateToMap(&(surf_point_in_->points[i]), &point_temp);
                    typename SurfFeatureMatch<_PointType>::SurfCostFactorInfo res;

                    if (surf_match_.Match(point_temp, res)) {
                        Eigen::Vector3d ori_point(surf_point_in_->points[i].x, 
                                                        surf_point_in_->points[i].y, 
                                                        surf_point_in_->points[i].z);
                        // ceres::CostFunction* cost_function = new se3PointSurfFactor(ori_point, res.norm_, res.D_);    
                        factor_container[i] = new se3PointSurfFactor(ori_point, res.norm_, res.D_);   
                        // problem.AddResidualBlock(cost_function, loss_function, parameters);
                    }
                    // 保存该点的匹配残差信息，匹配点信息 
                    if (save_match_info) {
                        points_registration_res_[surf_label_].residuals_[i] = res.residuals_;
                        points_registration_res_[surf_label_].nearly_points_[i] = std::move(res.matched_points_); 
                        // LOG(INFO) << "nearly_points_[i] size: "<< points_registration_res_[surf_label_].nearly_points_[i].size()
                        // <<", i:"<<i; 
                    }
                }
            ); 
            // LOG(INFO) << "points_registration_res_[surf_label_].nearly_points_ size: "
            // <<points_registration_res_.at(surf_label_).nearly_points_.size(); 
            //TicToc tt;    
            // 耗时很小   1~2ms左右
            for (uint32_t i = 0; i < surf_point_in_->points.size(); i++) {
                if (factor_container[i] != nullptr) {
                    problem.AddResidualBlock(factor_container[i], loss_function, parameters);
                    surf_num++;
                }
            }
            //tt.toc("AddResidualBlock ");
            //}
            //LOG(INFO) << "surf feature num:" << surf_num; 
            if (surf_num < 20) {
                printf("not enough surf points");
            }
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 
         * @details: 
         * @return {*}
         */        
        void addEdgeCostFactor(ceres::Problem& problem, ceres::LossFunction *loss_function,
                bool const& save_match_info) {
            if (edge_point_in_ == nullptr || edge_point_in_->empty()) return;  
            if (save_match_info) {
                points_registration_res_[edge_label_].residuals_.clear();  
                points_registration_res_[edge_label_].nearly_points_.clear();  
                points_registration_res_[edge_label_].residuals_.resize(edge_point_in_->points.size()); 
                points_registration_res_[edge_label_].nearly_points_.resize(edge_point_in_->points.size()); 
            }
            std::atomic<uint32_t> edge_num{0};
            std::vector<uint32_t> index(edge_point_in_->points.size());
            for (uint32_t i = 0; i < index.size(); ++i) {
                index[i] = i;
            }
            std::vector<ceres::CostFunction*> factor_container(surf_point_in_->points.size(), nullptr);
            //for (int i = 0; i < (int)edge_point_in_->points.size(); i++) {
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                _PointType point_in_ref;
                pointAssociateToMap(&(edge_point_in_->points[i]), &point_in_ref);
                typename EdgeFeatureMatch<_PointType>::EdgeCostFactorInfo res;
                // 在local map中搜索最近点 
                if (edge_match_.Match(point_in_ref, res)) {
                    Eigen::Vector3d ori_point(edge_point_in_->points[i].x, 
                                                                            edge_point_in_->points[i].y, 
                                                                            edge_point_in_->points[i].z);
                    // 传入变换前的点坐标以及匹配到的edge的两个点 
                    factor_container[i] = new se3PointEdgeFactor(ori_point, res.points_edge_[0], res.points_edge_[1]);    
                }
                // 保存该点的匹配残差信息，匹配点信息 
                if (save_match_info) {
                    points_registration_res_[surf_label_].residuals_[i] = res.residuals_;
                    points_registration_res_[surf_label_].nearly_points_[i] = std::move(res.matched_points_); 
                    // LOG(INFO) << "nearly_points_[i] size: "<< points_registration_res_[surf_label_].nearly_points_[i].size()
                    // <<", i:"<<i; 
                }
            }); 
            //LOG(INFO) << "edge feature num:" << edge_num; 
            if(edge_num < 20) {
                printf("not enough edge points");
            }
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void pointAssociateToMap(_PointType const *const pi, _PointType *const po) {
            Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
            Eigen::Vector3d point_w = q_w_curr * point_curr + t_w_curr;
            *po = *pi; 
            po->x = point_w.x();
            po->y = point_w.y();
            po->z = point_w.z();
            //po->intensity = 1.0;
        }

    private:
        std::string edge_label_, surf_label_;   // 线、面特征的标识
        // target pointcloud 
        typename pcl::PointCloud<_PointType>::ConstPtr surf_point_in_;
        typename pcl::PointCloud<_PointType>::ConstPtr edge_point_in_;
        // 匹配器 
        EdgeFeatureMatch<_PointType> edge_match_;
        SurfFeatureMatch<_PointType> surf_match_;
        // 匹配结果 
        RegistrationResult points_registration_res_;   
        double parameters[7] = {0, 0, 0, 1, 0, 0, 0};
        Eigen::Map<Eigen::Quaterniond> q_w_curr = Eigen::Map<Eigen::Quaterniond>(parameters);
        Eigen::Map<Eigen::Vector3d> t_w_curr = Eigen::Map<Eigen::Vector3d>(parameters + 4);
        uint16_t optimization_count_; 
}; // class LineSurfFeatureRegistration 
} // namespace 
}
