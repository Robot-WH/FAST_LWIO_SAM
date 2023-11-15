#pragma once

#include <ceres/ceres.h>
#include "lwio/Math.hpp"
#include "../parameters.hpp"

namespace lwio {
namespace estimator {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief 陀螺仪  预积分  
 * 
 */
class GyroPreIntegration {
public:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    GyroPreIntegration() = delete;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    GyroPreIntegration(const Eigen::Vector3d &linearized_bg)
        : linearized_bg_{linearized_bg}, jacobian_{Eigen::Matrix<double, 6, 6>::Identity()}, 
            sum_dt_{0.0}, delta_q_{Eigen::Quaterniond::Identity()} {
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void Reset() {
        curr_time_ = -1; 
        dt_buf_.clear();
        gyr_buf_.clear(); 
        delta_q_.setIdentity();
        jacobian_.setIdentity();
        sum_dt_ = 0.0;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief 
     * 
     * @param timestamp 
     * @param acc 
     * @param gyr 
     */
    void push_back(const double& timestamp, const Eigen::Vector3d &gyr) {
        if (curr_time_ < 0) {
            // 第一个imu数据 
            curr_time_ = timestamp;
            last_gyr_ = gyr; 
            first_gyr_ = last_gyr_;  
            return;  
        }
        double dt = timestamp - curr_time_;   
        curr_time_ = timestamp;
        dt_buf_.push_back(dt);
        gyr_buf_.push_back(gyr);
        propagate(dt, gyr);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief 
     * 
     * @param linearized_ba 
     * @param linearized_bg 
     */
    void repropagate(const Eigen::Vector3d &linearized_bg) {
        sum_dt_ = 0.0;
        last_gyr_ = first_gyr_;
        delta_q_.setIdentity();
        linearized_bg_ = linearized_bg;
        jacobian_.setIdentity();
        for (int i = 0; i < static_cast<int>(dt_buf_.size()); i++)
            propagate(dt_buf_[i],  gyr_buf_[i]);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief 
     * 
     * @param dt 
     * @param acc_0 
     * @param gyr_0 
     * @param acc_1 
     * @param gyr_1 
     * @param delta_p 
     * @param delta_q 
     * @param delta_v 
     * @param linearized_ba 
     * @param linearized_bg 
     * @param result_delta_p 
     * @param result_delta_q 
     * @param result_delta_v 
     * @param result_linearized_ba 
     * @param result_linearized_bg 
     * @param update_jacobian 
     */
    void midPointIntegration(double dt, const Eigen::Vector3d &gyr_0, const Eigen::Vector3d &gyr_1,
                            const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &linearized_bg,
                            Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_linearized_bg, 
                            bool update_jacobian) {
        Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + gyr_1) - linearized_bg;
        result_delta_q = delta_q * Eigen::Quaterniond(1, un_gyr(0) * dt / 2, un_gyr(1) * dt / 2, un_gyr(2) * dt / 2);
        result_linearized_bg = linearized_bg;  
        // 更新jacobian
        Eigen::MatrixXd F = Eigen::MatrixXd::Zero(6, 6);   

        Eigen::Matrix3d R_w_x;
        R_w_x<<0, -un_gyr(2), un_gyr(1),
            un_gyr(2), 0, -un_gyr(0),
            -un_gyr(1), un_gyr(0), 0;

        F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() - R_w_x * dt;
        F.block<3, 3>(0, 3) = -1.0 * Eigen::MatrixXd::Identity(3,3) * dt;
        F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();  
        jacobian_ = F * jacobian_;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief 
     * 
     * @param dt 
     * @param acc_1 
     * @param gyr_1 
     */
    void propagate(const double& dt, const Eigen::Vector3d &gyr_1) {
        Eigen::Quaterniond result_delta_q;
        Eigen::Vector3d result_linearized_bg;
        // std::cout << "midPointIntegration" << std::endl;
        midPointIntegration(dt, last_gyr_, gyr_1, delta_q_, linearized_bg_,
            result_delta_q, result_linearized_bg, 1);

        //checkJacobian(dt, acc_0_, last_gyr_, acc_1_, gyr_1_, delta_p, delta_q, delta_v,
        //                    linearized_ba_, linearized_bg_);
        delta_q_ = result_delta_q;
        linearized_bg_ = result_linearized_bg;
        delta_q_.normalize();
        sum_dt_ += dt;
        last_gyr_ = gyr_1;  
    }


    Eigen::Vector3d first_gyr_;
    Eigen::Vector3d linearized_bg_;

    Eigen::Matrix<double, 6, 6> jacobian_;

    double sum_dt_;
    Eigen::Quaterniond delta_q_;

    std::vector<double> dt_buf_;
    std::vector<Eigen::Vector3d> gyr_buf_;
private:
    double curr_time_ = -1;
    Eigen::Vector3d last_gyr_{-1, -1, -1};
};
}
}
