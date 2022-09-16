#pragma once 

#include "registration_base.hpp"
#include "ceres_factor/edge_factor.hpp"
#include "ceres_factor/surf_factor.hpp"
#include "Algorithm/Ceres/Parameterization/PoseSE3Parameterization.hpp"
#include "FeatureMatch/EdgeFeatureMatch.hpp"
#include "FeatureMatch/surfFeatureMatch.hpp"
#include "tic_toc.h"
#include <execution>  // C++ 17 并行算法 
#include <mutex>
#include <atomic>

namespace Slam3D {
namespace Algorithm {
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: 基于OptimizeMethod::GN/OptimizeMethod::LM法的边缘/面特征匹配  
 */    
template<typename _PointType>
class EdgeSurfFeatureRegistration : public RegistrationBase<_PointType> {
    private:
        #define handle_degenerate 0
        using SurfCostFactorInfo = typename SurfFeatureMatch<_PointType>::SurfCostFactorInfo;
        using EdgeCostFactorInfo = typename EdgeFeatureMatch<_PointType>::EdgeCostFactorInfo;
        using Base = RegistrationBase<_PointType>; 
        using RegistrationResult = typename Base::RegistrationResult;  
    public:
        enum class OptimizeMethod {GN, LM};
        struct OptionLM {
            uint16_t max_iterater_count_ = 30;   // 最大迭代优化次数   
        };
        struct Option {
            std::string edge_label_;
            std::string surf_label_;
            OptimizeMethod method_; 
            uint16_t max_iterater_count_ = 10;    // 初始优化迭代次数   每一次迭代需要重新找一次最近邻  
            uint16_t norm_iterater_count_ = 3;  // 常规优化次数
            OptionLM lm_option_;    // LM优化算法的设置 
        };  

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        EdgeSurfFeatureRegistration(Option option) : option_(option), 
            edge_match_(option.edge_label_), surf_match_(option.surf_label_) {
                if (option_.max_iterater_count_ < option_.norm_iterater_count_) {
                    option_.max_iterater_count_ = 10;
                    option_.norm_iterater_count_ = 3; 
                }
                optimization_count_ = option_.max_iterater_count_;    
                points_registration_res_.insert(std::make_pair(option_.edge_label_, typename Base::pointRegistrationResult())); 
                points_registration_res_.insert(std::make_pair(option_.surf_label_, typename Base::pointRegistrationResult())); 
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 传入
         * @details:  匹配逻辑是 target 与 source 匹配 
         * @param target_input  点云名称+点云数据 std::pair<std::string, PointCloudPtr> 
         */            
        void SetInputTarget(FeaturePointCloudContainer<_PointType> const& target_input) override {
            // TicToc tt;
            // tt.tic(); 
            if (target_input.find(option_.edge_label_) != target_input.end()) {
                edge_match_.SetSearchTarget(target_input.find(option_.edge_label_)->second);  
            }
            if (target_input.find(option_.surf_label_) != target_input.end()) {
                surf_match_.SetSearchTarget(target_input.find(option_.surf_label_)->second);  
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
            if (source_input.find(option_.edge_label_) != source_input.end()) {
                edge_point_in_ = source_input.find(option_.edge_label_)->second; 
            }
            if (source_input.find(option_.surf_label_) != source_input.end()) {
                surf_point_in_ = source_input.find(option_.surf_label_)->second; 
            }
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void SetMaxIteration(uint16_t const& n) override {
            if (n < option_.norm_iterater_count_) return;  
            option_.max_iterater_count_ = n;
            optimization_count_ = n;  
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void SetNormIteration(uint16_t const& n) override {
            if (n > option_.max_iterater_count_) return;  
            option_.norm_iterater_count_ = n;
            optimization_count_ = n;  
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 求解匹配 
         * @param[out] T 输入匹配预测初值   返回匹配的结果
         * @return {*}
         */            
        void Solve(Eigen::Isometry3d &T) override {
            TicToc tt;  
            tt.tic();  
            bool nearly_convergence = false;  // 连续两次收敛 才是真正的收敛 
            // 将预测位姿设定为优化前的初始值    
            q_w_l_ = Eigen::Quaterniond(T.rotation());
            t_w_l_ = T.translation();
            int iterCount = 0;
            bool save_match_info = false;     // 保存匹配信息 
            check_degenerate_ = false;  
            if (optimization_count_ > option_.norm_iterater_count_) {
                optimization_count_--;   
            }
            // 迭代
            for (iterCount = 0; iterCount < optimization_count_; iterCount++) {
                // LOG(INFO) << "--------------------------iterCount : "<< iterCount;  
                if (iterCount == optimization_count_ - 1) {
                    save_match_info = true;   // 最后一次迭代需要保存匹配的信息 
                }
                // 为每个特征构建残差factor  
                addSurfCostFactor(save_match_info);
                addEdgeCostFactor(save_match_info);  
                // scan-to-map优化
                // 对匹配特征点计算Jacobian矩阵，观测值为特征点到直线、平面的距离，
                // 构建高斯牛顿方程，迭代优化当前位姿，存transformTobeMapped
                if (option_.method_ == OptimizeMethod::GN) {
                    if (gnOptimization(iterCount) == true) {
                        break;              
                    }
                } else if (option_.method_ == OptimizeMethod::LM) {
                    if (lmOptimization(option_.lm_option_)) {
                        if (nearly_convergence) {
                            break;
                        } else {
                            nearly_convergence = true;  
                        }
                    } else {
                        nearly_convergence = false;  
                    }
                }
            }
            T.linear() = q_w_l_.toRotationMatrix();
            T.translation() = t_w_l_;
            // tt.toc("solve");  
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 获取本次匹配的结果信息  
         * @details: 即每一点的匹配残差以及匹配点 
         * @return {*}
         */        
        RegistrationResult const& GetRegistrationResult() const override {
            return points_registration_res_; 
        }

    protected:

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // 加速前：约51ms  
        // 加速后：约33ms 
        void addSurfCostFactor(bool const& save_match_info) {
            if (surf_point_in_ == nullptr || surf_point_in_->empty()) return;  
            // 匹配结束时  保存每一个点的匹配信息  - 匹配点 + 残差  
            if (save_match_info) {
                points_registration_res_[option_.surf_label_].residuals_.clear();  
                points_registration_res_[option_.surf_label_].nearly_points_.clear();  
                points_registration_res_[option_.surf_label_].residuals_.resize(surf_point_in_->points.size()); 
                points_registration_res_[option_.surf_label_].nearly_points_.resize(surf_point_in_->points.size()); 
            }
            //std::cout<<"addSurfCostFactor, points.size(): "<<(int)surf_point_in_->points.size()<<std::endl;
            std::vector<uint32_t> index(surf_point_in_->points.size());
            for (uint32_t i = 0; i < index.size(); ++i) {
                index[i] = i;
            }
            surf_num_ = 0;  
            origin_surf_points_.clear();
            surf_matched_info_.clear();
            surf_matched_info_.resize(surf_point_in_->points.size()); 
            origin_surf_points_.resize(surf_point_in_->points.size());
            //for (int i = 0; i < (int)surf_point_in_->points.size(); i++) {
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), 
                [&](const size_t &i) {
                    _PointType point_temp;
                    pointLocalToMap(&(surf_point_in_->points[i]), &point_temp);
                    typename SurfFeatureMatch<_PointType>::SurfCostFactorInfo res;

                    if (surf_match_.Match(point_temp, res)) {
                        Eigen::Vector3d ori_point(surf_point_in_->points[i].x, 
                                                        surf_point_in_->points[i].y, 
                                                        surf_point_in_->points[i].z);
                        origin_surf_points_[i] = ori_point;  
                        surf_matched_info_[i] = res;  
                        surf_num_++; 
                    }
                    // 保存该点的匹配残差信息，匹配点信息  不论是否匹配成功  
                    if (save_match_info) {
                        points_registration_res_[option_.surf_label_].residuals_[i] = res.residuals_;
                        points_registration_res_[option_.surf_label_].nearly_points_[i] = std::move(res.matched_points_); 
                        // LOG(INFO) << "nearly_points_[i] size: "<< points_registration_res_[surf_label_].nearly_points_[i].size()
                        // <<", i:"<<i; 
                    }
                }
            ); 
            //LOG(INFO) << "surf feature num:" << surf_num_; 
            if (surf_num_ < 20) {
                printf("not enough surf points");
            }
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void addEdgeCostFactor(bool const& save_match_info) {
            if (edge_point_in_ == nullptr || edge_point_in_->empty()) return;  
            if (save_match_info) {
                points_registration_res_[option_.edge_label_].residuals_.clear();  
                points_registration_res_[option_.edge_label_].nearly_points_.clear();  
                points_registration_res_[option_.edge_label_].residuals_.resize(edge_point_in_->points.size()); 
                points_registration_res_[option_.edge_label_].nearly_points_.resize(edge_point_in_->points.size()); 
            }
            std::vector<uint32_t> index(edge_point_in_->points.size());
            for (uint32_t i = 0; i < index.size(); ++i) {
                index[i] = i;
            }
            edge_num_ = 0;  
            origin_edge_points_.clear();
            edge_matched_info_.clear();
            edge_matched_info_.resize(edge_point_in_->points.size()); 
            origin_edge_points_.resize(edge_point_in_->points.size());
            //for (int i = 0; i < (int)edge_point_in_->points.size(); i++) {
            std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                _PointType point_in_ref;
                pointLocalToMap(&(edge_point_in_->points[i]), &point_in_ref);
                typename EdgeFeatureMatch<_PointType>::EdgeCostFactorInfo res;
                // 在local map中搜索最近点 
                if (edge_match_.Match(point_in_ref, res)) {
                    Eigen::Vector3d ori_point(edge_point_in_->points[i].x, 
                                                                            edge_point_in_->points[i].y, 
                                                                            edge_point_in_->points[i].z);
                    origin_edge_points_[i] = ori_point;  
                    edge_matched_info_[i] = res;  
                    edge_num_++; 
                }
                // 保存该点的匹配残差信息，匹配点信息 
                if (save_match_info) {
                    points_registration_res_[option_.edge_label_].residuals_[i] = res.residuals_;
                    points_registration_res_[option_.edge_label_].nearly_points_[i] = std::move(res.matched_points_); 
                }
            }); 
            //LOG(INFO) << "edge feature num:" << edge_num_; 
            if(edge_num_ < 20) {
                printf("not enough edge points");
            }
        }
        
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void makeHessian(Eigen::Matrix<double, 6, 6>& JTJ, Eigen::Matrix<double, 6, 1>& JTR) {
            Eigen::MatrixXd J(Eigen::MatrixXd::Zero(edge_num_ + surf_num_, 6)); // 残差关于状态的jacobian矩阵
            Eigen::MatrixXd R(Eigen::MatrixXd::Zero(edge_num_ + surf_num_, 1));   
            JTR.setZero();
            JTJ.setZero();   

            Eigen::Vector3d grad, pointOri;  
            float residual = 0;  
            int edge_point_num = edge_matched_info_.size(); 
            int surf_point_num = surf_matched_info_.size(); 
            int all_point_num = surf_point_num + edge_point_num; 
            int valid_count = 0; 
            // 遍历匹配特征点，构建Jacobian矩阵
            for (int i = 0; i < all_point_num; i++) {
                if (i < edge_point_num) {
                    if (!edge_matched_info_[i].is_valid_) {
                        continue; 
                    }
                    // 激光系下点的坐标 
                    pointOri = origin_edge_points_[i];
                    // 残差以及其梯度向量  
                    grad = edge_matched_info_[i].norm_; 
                    residual = edge_matched_info_[i].residuals_;
                } else {
                    int surf_ind = i - edge_point_num;
                    if (!surf_matched_info_[surf_ind].is_valid_) {
                        continue; 
                    }
                    // 激光系下点的坐标 
                    pointOri = origin_surf_points_[surf_ind];
                    // 残差以及其梯度向量  
                    grad = surf_matched_info_[surf_ind].norm_; 
                    residual = surf_matched_info_[surf_ind].residuals_;
                }

                Eigen::Matrix<double, 3, 6> d_P_T;
                // 左乘扰动 
                //d_P_T.block<3, 3>(0, 0) = -Math::GetSkewMatrix<double>(q_w_l_ * pointOri);    // 关于旋转
                //  右乘扰动  
                d_P_T.block<3, 3>(0, 0) = (-q_w_l_.toRotationMatrix() 
                                                                        * Math::GetSkewMatrix<double>(pointOri));
                d_P_T.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();     // 关于平移  
                Eigen::Matrix<double, 1, 3> d_r_P = grad.transpose();  
                J.block<1, 6>(valid_count, 0) = d_r_P * d_P_T;  // lidar -> camera
                R(valid_count, 0) = residual; // 点到直线距离、平面距离，作为观测值
                valid_count++; 
            }
            JTJ = J.transpose() * J;
            JTR = J.transpose() * R;
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void checkDegenerate(Eigen::Matrix<double, 6, 6> const& hessian) {
            //  对 ATA进行特征分解 
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(hessian);      // cov 是SelfAdjoint
            Eigen::MatrixXd V = eigensolver.eigenvectors();
            Eigen::MatrixXd V2 = V;
            
            is_degenerate_ = false;
            float degeneracy_thresh = 100;   // 特征值阈值
            // 从最小特征
            for (int i = 5; i >= 0; i--) {   // 将小于阈值的特征值对应的特征向量设为0  
                if (eigensolver.eigenvalues()[i] < degeneracy_thresh) {
                    V2.row(i) = Eigen::MatrixXd::Zero(1, 6);
                    is_degenerate_ = true;
                    std::cout<<common::RED<<"is_degenerate_ !!!, eigensolver: "<<
                    eigensolver.eigenvalues()[i]<<std::endl;
                } else {
                    break;
                }
            }
            Map_ = V.inverse() * V2;              
        }
        
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 四元数 + xyz 优化   
         * @details: 
         * @param {int} iterCount
         * @return {*}
         */    
        bool gnOptimization(int iterCount) {
            // 当前帧匹配特征点数太少
            if (edge_num_ + surf_num_ < 10) {
                std::cout<<common::YELLOW<<"not enough feature, num: "
                <<edge_num_ + surf_num_<<common::RESET<< std::endl;
                return false;
            }
            // Eigen::MatrixXd X(Eigen::MatrixXd::Zero(6, 1));   
            Eigen::Matrix<double, 6, 6> JTJ;
            Eigen::Matrix<double, 6, 1> JTR;
            makeHessian(JTJ, JTR);
            Eigen::Matrix<double, 6, 1> delta_x = JTJ.colPivHouseholderQr().solve(-JTR);    // QR分解 解AX=B 
            // tt.toc("QR solve ");
            // // J^T·J·delta_x = -J^T·f 高斯牛顿
            // X = JTJ.ldlt().solve(-JTR);    //  方式2：采用LDL
            // tt.toc("LDLT solve ");
            #if handle_degenerate
                if (iterCount == 0) {
                    checkDegenerate(JTJ); 
                }
                if (is_degenerate_) {
                        delta_x = Map_ * delta_x;
                }
            #endif
            updateState(delta_x);
            double deltaR = sqrt(pow(delta_x(0, 0), 2) +
                                                        pow(delta_x(1, 0), 2) +
                                                        pow(delta_x(2, 0), 2)); 
            double deltaT = sqrt(pow(delta_x(3, 0) * 100, 2) 
                                                    + pow(delta_x(4, 0) * 100, 2) 
                                                    + pow(delta_x(5, 0) * 100, 2)); // unit:cm  
            // delta_x很小，认为收敛   
            if (deltaR < 0.0009 && deltaT < 0.05) {   // 角度变化 < 0.05度  位置变化 < 0.05 cm 
                return true; 
            }
            return false; 
        }
        
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 使用OptimizeMethod::LM法进行求解 
         * @details 四元数 + xyz 优化   
         * @return {*}
         */    
        bool lmOptimization(OptionLM const& option) {
            // 当前帧匹配特征点数太少
            if (edge_num_ + surf_num_ < 10) {
                std::cout<<common::YELLOW<<"not enough feature, num: "
                <<edge_num_ + surf_num_<<common::RESET<< std::endl;
                return false;
            }
            // 计算初始残差  
            double lastChi = calcResidual();  
            bool init = false; 
            double current_lambda = 0;     // 阻尼系数
            double ni = 2;  // 当下降不好时   对于阻尼的放大系数  
            // 迭代n次 
            for (int num = 0; num < option.max_iterater_count_; num++) {
                Eigen::Matrix<double, 6, 6> JTJ;
                Eigen::Matrix<double, 6, 1> JTR;
                makeHessian(JTJ, JTR);
                bool oneStepSuccess = false;
                int false_cnt = 0;
                // 循环直到找到正确的step
                while (!oneStepSuccess && false_cnt < 5) { 
                    //不断尝试 Lambda, 直到成功迭代一步
                    if (!init) {
                        // 初始化阻尼系数  
                        double diag_max_value = 0;
                        assert(JTJ.rows() == JTJ.cols() && "Hessian is not square");
                        int size = JTJ.cols();
                        // 找出Hessian 对角线最大的元素
                        for (int i = 0; i < size; ++i) {
                            diag_max_value = std::max(fabs(JTJ(i, i)), diag_max_value);
                        }
                        // 系数
                        double tau = 1e-4;    
                        // 乘上系数    作为初始的阻尼系数
                        current_lambda = tau * diag_max_value;
                        init = true;
                    }
                    Eigen::Matrix<double, 6, 6> H = JTJ + current_lambda * Eigen::Matrix<double, 6, 6>::Identity();  
                    Eigen::Matrix<double, 6, 1> delta_x = H.ldlt().solve(-JTR);     // 正定对称  采用LDL
                    // 根据增量大小判断是否迭代停止  
                    double deltaR = sqrt(pow(delta_x(0, 0), 2) 
                                                            + pow(delta_x(1, 0), 2) 
                                                            + pow(delta_x(2, 0), 2)); 
                    double deltaT = sqrt(pow(delta_x(3, 0) * 100, 2) 
                                                            + pow(delta_x(4, 0) * 100, 2) 
                                                            + pow(delta_x(5, 0) * 100, 2)); // unit:cm  
                    // delta_x很小，认为收敛   
                    if (deltaR < 0.0009 && deltaT < 0.05) {   // 角度变化 < 0.05度  位置变化 < 0.05 cm
                        // LOG(INFO) << "LM converge";
                        return true; 
                    }
                    // #if handle_degenerate
                    //     if (check_degenerate_) {
                    //         checkDegenerate(JTJ); 
                    //         check_degenerate_ = true;  
                    //     }
                    //     if (is_degenerate_) {
                    //         X = Map_ * X;
                    //     }
                    // #endif
                    Eigen::Vector3d old_t_w_l = t_w_l_;
                    Eigen::Quaterniond old_q_w_l = q_w_l_;
                    updateState(delta_x);// 更新当前位姿 x = x + delta_x
                    double currChi = updateResidual();     // 求解当前状态更新后的残差 
                    // 计算线性近似的下降值   L(0) - L(delta_x)
                    double linear_approximation = 0;
                    linear_approximation = delta_x.transpose() * (current_lambda * delta_x - JTR);
                    linear_approximation += 1e-6;    // make sure it's non-zero :)        
                    double rho = (lastChi - currChi) / linear_approximation;
                    
                    if (rho > 0) {
                        // 下降了
                        // 当 rho < 0.5 时  此时 线性化不好，alpha > 1 增大阻尼   
                        // 当rho > 0.5时 ，线性化可以接受，alpha < 1 , 逐渐的减少阻尼 ，最多减少 3倍
                        double alpha = 1. - pow((2 * rho - 1), 3);   // alpha 最大为2 
                        // alpha = std::min(alpha, 2. / 3.);
                        double scaleFactor = (std::max)(1. / 3., alpha); 
                        current_lambda *= scaleFactor;
                        ni = 2;
                        oneStepSuccess = true;  
                        lastChi = currChi; 
                    } else {
                        // 非下降方向则拒绝更新  增大阻尼  
                        t_w_l_ = old_t_w_l;
                        q_w_l_ = old_q_w_l;  
                        false_cnt++; 
                        current_lambda *= ni;   
                        ni *= 2;
                    }
                }
                // 如果 本次迭代 找不到正确的step 
                if (!current_lambda) {
                    return true;  
                }
            }
            return false; 
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        double calcResidual() {
            double residual = 0;  
            int edge_point_num = edge_matched_info_.size(); 
            int surf_point_num = surf_matched_info_.size(); 
            int all_point_num = surf_point_num + edge_point_num; 
            /**
             * @todo 使用多线程加速
             */
            for (int i = 0; i < all_point_num; i++) {
                if (i < edge_point_num) {
                    if (!edge_matched_info_[i].is_valid_) {
                        continue; 
                    }
                    residual += pow(edge_matched_info_[i].residuals_, 2);
                } else {
                    int surf_ind = i - edge_point_num;
                    if (!surf_matched_info_[surf_ind].is_valid_) {
                        continue; 
                    }
                    residual += pow(surf_matched_info_[surf_ind].residuals_, 2);
                }
            }
            return residual;  
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 更新姿态变化后特征的残差
         */        
        double updateResidual() {
            // 计算更新状态后的残差 
            int edge_point_num = edge_matched_info_.size(); 
            int surf_point_num = surf_matched_info_.size(); 
            double curr_residual = 0.; 
            // 遍历匹配数据 
            for (int i = 0; i < surf_point_num; i++) {
                if (!surf_matched_info_[i].is_valid_) {
                    continue; 
                }
                // 激光系下点的坐标 
                Eigen::Vector3d point_in_map = q_w_l_ * origin_surf_points_[i] + t_w_l_;  
                // 平面参数 
                Eigen::Vector3d& norm_ = surf_matched_info_[i].norm_; 
                double& D_ = surf_matched_info_[i].D_;  
                // 更新残差  
                surf_matched_info_[i].residuals_ = norm_.dot(point_in_map) + D_;
                curr_residual += pow(surf_matched_info_[i].residuals_, 2); 
            }
            return curr_residual;  
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void updateState(Eigen::Matrix<double, 6, 1> const& X) {
            // 更新当前位姿 x = x + delta_x
            t_w_l_[0] += X(3, 0);
            t_w_l_[1] += X(4, 0);
            t_w_l_[2] += X(5, 0);
            // 转四元数增量 
            Eigen::Vector3d delta_rot = {X(0, 0), X(1, 0), X(2, 0)};
            Eigen::AngleAxisd delta_rot_v(delta_rot.norm() / 2, delta_rot.normalized());
            Eigen::Quaterniond delta_q(delta_rot_v);  
            // 更新旋转 
            // q_w_l_ = delta_q * q_w_l_;   // 左乘扰动采用左乘进行更新
            q_w_l_ = q_w_l_ * delta_q;     // 右乘扰动采用右乘进行更新
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void pointLocalToMap(_PointType const *const p_l, _PointType *const p_w) {
            Eigen::Vector3d point_curr(p_l->x, p_l->y, p_l->z);
            Eigen::Vector3d point_w = q_w_l_ * point_curr + t_w_l_;
            *p_w = *p_l; 
            p_w->x = point_w.x();
            p_w->y = point_w.y();
            p_w->z = point_w.z();
        }

    private:
        Option option_;  
        std::vector<Eigen::Vector3d> origin_surf_points_;
        std::vector<Eigen::Vector3d> origin_edge_points_;
        std::vector<SurfCostFactorInfo> surf_matched_info_;
        std::vector<EdgeCostFactorInfo> edge_matched_info_;
        // 匹配器 
        EdgeFeatureMatch<_PointType> edge_match_;
        SurfFeatureMatch<_PointType> surf_match_;
        // target pointcloud 
        typename pcl::PointCloud<_PointType>::ConstPtr surf_point_in_ = nullptr;
        typename pcl::PointCloud<_PointType>::ConstPtr edge_point_in_ = nullptr;
        // 求解结果
        RegistrationResult points_registration_res_;   
        Eigen::Quaterniond q_w_l_;
        Eigen::Vector3d t_w_l_;
        Eigen::MatrixXd Map_;     //  退化时，对于X的remapping 矩阵

        uint16_t optimization_count_ = 0;
        std::atomic<uint16_t> edge_num_{0};
        std::atomic<uint16_t> surf_num_{0};

        bool is_degenerate_ = false; 
        bool check_degenerate_ = false; 
}; // class LineSurfFeatureRegistration 
}
}

