/*
 * @Copyright(C): Your Company
 * @FileName: 文件名
 * @Author: 作者
 * @Version: 版本
 * @Date: 2022-03-11 12:46:36
 * @Description: 
 * @Others: 
 */

#pragma once 

#include <eigen3/Eigen/Dense>

namespace Math {

// 获取反对称矩阵 
template<typename T>
static Eigen::Matrix<T, 3, 3> GetSkewMatrix(const Eigen::Matrix<T, 3, 1>& v) {
    Eigen::Matrix<T, 3, 3> w;
    w <<  0.,   -v(2, 0),  v(1, 0),
        v(2, 0),  0.,   -v(0, 0),
        -v(1, 0),  v(0, 0),  0.;
    return w;
}

// 李代数转四元数 + XYZ 
static void GetTransformFromSe3(const Eigen::Matrix<double,6,1>& se3, 
                                                                        Eigen::Quaterniond& q, Eigen::Vector3d& t) {
    Eigen::Vector3d omega(se3.data());
    Eigen::Vector3d upsilon(se3.data()+3);
    Eigen::Matrix3d Omega = GetSkewMatrix(omega);

    double theta = omega.norm();
    double half_theta = 0.5 * theta;

    double imag_factor;
    double real_factor = cos(half_theta);       // 四元数实部 

    if (theta < 1e-10) {  
        // sin( theta / 2)泰勒展开  
        double theta_sq = theta * theta;
        double theta_po4 = theta_sq * theta_sq;
        imag_factor = 0.5-0.0208333 * theta_sq+0.000260417 * theta_po4;       // 同时除了 theta
    } else {
        double sin_half_theta = sin(half_theta);
        imag_factor = sin_half_theta / theta;
    }

    q = Eigen::Quaterniond(real_factor, 
                                                    imag_factor * omega.x(), 
                                                    imag_factor * omega.y(), 
                                                    imag_factor * omega.z());

    Eigen::Matrix3d J;
    if (theta < 1e-10) {
        J = q.matrix();
    } else {
        //  罗德里格斯 
        Eigen::Matrix3d Omega2 = Omega * Omega;
        J = (Eigen::Matrix3d::Identity() 
                + (1 - cos(theta)) / (theta * theta) * Omega + (theta - sin(theta)) / (pow(theta,3)) * Omega2);
    }
    t = J*upsilon;
}

template <typename Derived>
static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta) {
    typedef typename Derived::Scalar Scalar_t;

    Eigen::Quaternion<Scalar_t> dq;
    Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
    half_theta /= static_cast<Scalar_t>(2.0);
    dq.w() = static_cast<Scalar_t>(1.0);
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
}

/**
 * @brief:  四元数 左乘矩阵
 * @param {Quaterniond const&} q
 * @return {*}
 */
static Eigen::Matrix4d QLeft(Eigen::Quaterniond const& q) {
    Eigen::Matrix4d m;
    m << q.w(), -q.x(), -q.y(), -q.z(),
                q.x(), q.w(), -q.z(), q.y(),
                q.y(), q.z(), q.w(), -q.x(),
                q.z(), -q.y(), q.x(), q.w();  
    return m;  
}

/**
 * @brief:  四元数 右乘矩阵
 * @param {Quaterniond const&} q
 * @return {*}
 */
static Eigen::Matrix4d QRight(Eigen::Quaterniond const& q) {
    Eigen::Matrix4d m;
    m << q.w(), -q.x(), -q.y(), -q.z(),
                q.x(), q.w(), q.z(), -q.y(),
                q.y(), -q.z(), q.w(), q.x(),
                q.z(), q.y(), -q.x(), q.w();  
    return m;  
}

static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R) {
    Eigen::Vector3d n = R.col(0);
    Eigen::Vector3d o = R.col(1);
    Eigen::Vector3d a = R.col(2);

    Eigen::Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr / M_PI * 180.0;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> &ypr) {
    typedef typename Derived::Scalar Scalar_t;

    Scalar_t y = ypr(0) / 180.0 * M_PI;
    Scalar_t p = ypr(1) / 180.0 * M_PI;
    Scalar_t r = ypr(2) / 180.0 * M_PI;

    Eigen::Matrix<Scalar_t, 3, 3> Rz;
    Rz << cos(y), -sin(y), 0,
        sin(y), cos(y), 0,
        0, 0, 1;

    Eigen::Matrix<Scalar_t, 3, 3> Ry;
    Ry << cos(p), 0., sin(p),
        0., 1., 0.,
        -sin(p), 0., cos(p);

    Eigen::Matrix<Scalar_t, 3, 3> Rx;
    Rx << 1., 0., 0.,
        0., cos(r), -sin(r),
        0., sin(r), cos(r);

    return Rz * Ry * Rx;
}

/**
 * @brief 一元二次方程求解 
 * 
 * @tparam T 
 * @param a 
 * @param b 
 * @param c 
 * @param x1 
 * @param x2 
 * @return true 
 * @return false 
 */
template <typename T>
static bool solveQuadraticEquation(const T& a, const T& b, const T& c, T& x1, T& x2) {
    if (fabs(a) < 1e-12) {
        x1 = x2 = -c / b;
        return true;
    }
    T delta2 = b * b - 4.0 * a * c;

    if (delta2 == 0) {
        x1 = (-b) / (2.0 * a);
        x2 = (-b) / (2.0 * a);
        return true;  
    }

    if (delta2 < 0.0) {
        // std::cout << "delta2: " << delta2 << std::endl;
        // if (delta2 < -0.001) {
            return false;
        // }
    }
    T delta = sqrt(delta2);
    x1 = (-b + delta) / (2.0 * a);
    x2 = (-b - delta) / (2.0 * a);
    return true;
}


} // namespace Math
