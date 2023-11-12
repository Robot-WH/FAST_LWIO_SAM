#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "lwio/Math.hpp"

class PoseLocalParameterization : public ceres::LocalParameterization {
public:    

/**
 * @brief 扰动是对平移和旋转分别添加扰动 
 * 
 * @param x 平移 + 四元数  
 * @param delta 平移增量 + so3轴角变化 
 * @param x_plus_delta 
 * @return true 
 * @return false 
 */
virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const {
    // 前3为是平移   后4维是四元数 
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);      // 传入序列 (x, y, z, w)构造eigen 四元数   
    
    Eigen::Map<const Eigen::Vector3d> dp(delta);
    // so3 转换 四元数  
    Eigen::Quaterniond dq = Math::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    p = _p + dp;
    q = (_q * dq).normalized();

    return true;
}

/**
 * @brief 李群对李代数的jacobian
 * 
 * @param x 
 * @param jacobian 
 * @return true 
 * @return false 
 */
virtual bool ComputeJacobian(const double *x, double *jacobian) const {
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}
/**
 * @brief  维护的状态维度      XYZ + 四元数  
 * 
 * @return int 
 */
virtual int GlobalSize() const { return 7;};

/**
 * @brief 线性化，计算jacobian时使用的状态维度  XYZ + so3 轴角，这也是优化后求解出来的增量  
 * 
 * @return int 
 */
virtual int LocalSize() const { return 6; };
};
