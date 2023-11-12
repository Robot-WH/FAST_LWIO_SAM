#pragma once
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

#include "../parameters.hpp"
#include "lwio/Math.hpp"

namespace lwio {
namespace estimator {

/**
 * @brief 里程计残差因子  
 * @param 6: 残差的维度  只有平移和旋转残差
 *                     7： 第i个位姿状态的维度  xyz + 旋转四元数 
 *                      7： 第j个位姿状态的维度  xyz + 旋转四元数 
 */
class OdomFactor : public ceres::SizedCostFunction<6, 7, 7> {
  public:
    OdomFactor() = delete;
    OdomFactor(const Eigen::Isometry3d& odom, const Eigen::Matrix<double, 6, 6>& cov) 
    : delta_P_(odom.translation()), delta_Q_(odom.linear()), cov_(cov) {
    }

    /**
     * @brief 手动求导   ——  在这里实现jacobian 和 残差的计算
     * 
     * @param parameters 传入的状态由AddResidualBlock()设置
     * @param residuals 
     * @param jacobians 
     * @return true 
     * @return false 
     */
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
        // 这里parameters即由传入的状态由AddResidualBlock()设置
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

        // 残差更新  
        Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
        residual.setZero();
        residual.block<3, 1>(0, 0) = Qi * delta_P_ + Pi - Pj;
        residual.block<3, 1>(3, 0) = 2 * (delta_Q_.inverse() * Qi.inverse() * Qj).vec();

        Eigen::Matrix<double, 6, 6> sqrt_info = cov_;
        // sqrt_info = sqrt(cov_^-1)^T
        sqrt_info = Eigen::LLT<Eigen::Matrix<double, 6, 6>>(sqrt_info.inverse()).matrixL().transpose();
        
        residual = sqrt_info * residual;

        if (jacobians) {
            // 残差关于i位姿的jacobian   
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
                jacobian_pose_i.setZero();
                jacobian_pose_i.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
                jacobian_pose_i.block<3, 3>(0, 3) = -(Qi.toRotationMatrix() * Math::GetSkewMatrix(delta_P_)); 

                jacobian_pose_i.block<3, 3>(3, 3) = 
                    -(Math::QLeft(Qj.inverse() * Qi) * Math::QRight(delta_Q_)).bottomRightCorner<3, 3>();

                jacobian_pose_i = sqrt_info * jacobian_pose_i;    // 和信息矩阵耦合  

                if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8) {
                    // ROS_WARN("numerical unstable in preintegration");
                    //std::cout << sqrt_info << std::endl;
                    //ROS_BREAK();
                }
            }
            // 残差关于i + 1位姿的jacobian   
            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
                jacobian_pose_j.setZero();
                jacobian_pose_j.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();

                jacobian_pose_j.block<3, 3>(0, 3) = 
                    Math::QLeft(delta_Q_.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();

                jacobian_pose_j = sqrt_info * jacobian_pose_j;    // 和信息矩阵耦合  

                if (jacobian_pose_j.maxCoeff() > 1e8 || jacobian_pose_j.minCoeff() < -1e8) {
                    // ROS_WARN("numerical unstable in preintegration");
                    //std::cout << sqrt_info << std::endl;
                    //ROS_BREAK();
                }
            }
        }
        return true;
    }

    //bool Evaluate_Direct(double const *const *parameters, Eigen::Matrix<double, 15, 1> &residuals, Eigen::Matrix<double, 15, 30> &jacobians);

    //void checkCorrection();
    //void checkTransition();
    //void checkJacobian(double **parameters);
    Eigen::Quaterniond delta_Q_;
    Eigen::Vector3d delta_P_;  
    Eigen::Matrix<double, 6, 6> cov_;
};
}
}
