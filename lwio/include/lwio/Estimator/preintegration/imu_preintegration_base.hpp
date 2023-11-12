#pragma once

#include <ceres/ceres.h>
#include "lwio/Math.hpp"
#include "../parameters.hpp"

namespace lwio {
namespace estimator {

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief IMU 预积分  
 * 
 */
class ImuPreIntegrationBase {
public:
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ImuPreIntegrationBase() = delete;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ImuPreIntegrationBase(double ACC_N, double GYR_N, double ACC_W, double GYR_W,
            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg)
        : ACC_N_{ACC_N}, GYR_N_{GYR_N}, ACC_W_{ACC_W}, GYR_W_{GYR_W},
            linearized_ba_{linearized_ba}, linearized_bg_{linearized_bg},
            jacobian{Eigen::Matrix<double, 15, 15>::Identity()}, covariance_{Eigen::Matrix<double, 15, 15>::Zero()},
            sum_dt_{0.0}, delta_p_{Eigen::Vector3d::Zero()}, delta_q_{Eigen::Quaterniond::Identity()}, 
            delta_v_{Eigen::Vector3d::Zero()} {
        // 控制输入噪声协方差矩阵
        noise_ = Eigen::Matrix<double, 18, 18>::Zero();
        noise_.block<3, 3>(0, 0) =  (ACC_N_ * ACC_N_) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(3, 3) =  (GYR_N_ * GYR_N_) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(6, 6) =  (ACC_N_ * ACC_N_) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(9, 9) =  (GYR_N_ * GYR_N_) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(12, 12) =  (ACC_W_ * ACC_W_) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(15, 15) =  (GYR_W_ * GYR_W_) * Eigen::Matrix3d::Identity();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief 
     * 
     * @param timestamp 
     * @param acc 
     * @param gyr 
     */
    void push_back(const double& timestamp, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr) {
        if (curr_time_ < 0) {
            // 第一个imu数据 
            curr_time_ = timestamp;
            acc_0_ = acc;
            gyr_0_ = gyr; 
            return;  
        }
        double dt = timestamp - curr_time_;   
        curr_time_ = timestamp;
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        propagate(dt, acc, gyr);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief 
     * 
     * @param linearized_ba 
     * @param linearized_bg 
     */
    void repropagate(const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg) {
        sum_dt_ = 0.0;
        acc_0_ = linearized_acc;
        gyr_0_ = linearized_gyr;
        delta_p_.setZero();
        delta_q_.setIdentity();
        delta_v_.setZero();
        linearized_ba_ = linearized_ba;
        linearized_bg_ = linearized_bg;
        jacobian.setIdentity();
        covariance_.setZero();
        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
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
    void midPointIntegration(double dt, 
                            const Eigen::Vector3d &acc_0, const Eigen::Vector3d &gyr_0,
                            const Eigen::Vector3d &acc_1, const Eigen::Vector3d &gyr_1,
                            const Eigen::Vector3d &delta_p, const Eigen::Quaterniond &delta_q, const Eigen::Vector3d &delta_v,
                            const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                            Eigen::Vector3d &result_delta_p, Eigen::Quaterniond &result_delta_q, Eigen::Vector3d &result_delta_v,
                            Eigen::Vector3d &result_linearized_ba, Eigen::Vector3d &result_linearized_bg, bool update_jacobian) {
        //ROS_INFO("midpoint integration");
        Eigen::Vector3d un_acc_0 = delta_q * (acc_0 - linearized_ba);
        Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + gyr_1) - linearized_bg;
        result_delta_q = delta_q * Eigen::Quaterniond(1, un_gyr(0) * dt / 2, un_gyr(1) * dt / 2, un_gyr(2) * dt / 2);
        Eigen::Vector3d un_acc_1 = result_delta_q * (acc_1 - linearized_ba);
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        result_delta_p = delta_p + delta_v * dt + 0.5 * un_acc * dt * dt;
        result_delta_v = delta_v + un_acc * dt;
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;         

        if(update_jacobian) {
            Eigen::Vector3d w_x = 0.5 * (gyr_0 + gyr_1) - linearized_bg;
            Eigen::Vector3d a_0_x = acc_0 - linearized_ba;
            Eigen::Vector3d a_1_x = acc_1 - linearized_ba;
            Eigen::Matrix3d R_w_x, R_a_0_x, R_a_1_x;

            R_w_x<<0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
            R_a_0_x<<0, -a_0_x(2), a_0_x(1),
                a_0_x(2), 0, -a_0_x(0),
                -a_0_x(1), a_0_x(0), 0;
            R_a_1_x<<0, -a_1_x(2), a_1_x(1),
                a_1_x(2), 0, -a_1_x(0),
                -a_1_x(1), a_1_x(0), 0;

            Eigen::MatrixXd F = Eigen::MatrixXd::Zero(15, 15);
            F.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * dt * dt + 
                                    -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * dt) * dt * dt;
            F.block<3, 3>(0, 6) = Eigen::MatrixXd::Identity(3,3) * dt;
            F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * dt * dt;
            F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * dt * dt * -dt;
            F.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() - R_w_x * dt;
            F.block<3, 3>(3, 12) = -1.0 * Eigen::MatrixXd::Identity(3,3) * dt;
            F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * dt + 
                                    -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Eigen::Matrix3d::Identity() - R_w_x * dt) * dt;
            F.block<3, 3>(6, 6) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * dt;
            F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * dt * -dt;
            F.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();
            F.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();
            //cout<<"A"<<endl<<A<<endl;

            Eigen::MatrixXd V = Eigen::MatrixXd::Zero(15,18);
            V.block<3, 3>(0, 0) =  0.25 * delta_q.toRotationMatrix() * dt * dt;
            V.block<3, 3>(0, 3) =  0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x  * dt * dt * 0.5 * dt;
            V.block<3, 3>(0, 6) =  0.25 * result_delta_q.toRotationMatrix() * dt * dt;
            V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3) =  0.5 * Eigen::MatrixXd::Identity(3,3) * dt;
            V.block<3, 3>(3, 9) =  0.5 * Eigen::MatrixXd::Identity(3,3) * dt;
            V.block<3, 3>(6, 0) =  0.5 * delta_q.toRotationMatrix() * dt;
            V.block<3, 3>(6, 3) =  0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x  * dt * 0.5 * dt;
            V.block<3, 3>(6, 6) =  0.5 * result_delta_q.toRotationMatrix() * dt;
            V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
            V.block<3, 3>(9, 12) = Eigen::MatrixXd::Identity(3,3) * dt;
            V.block<3, 3>(12, 15) = Eigen::MatrixXd::Identity(3,3) * dt;

            // step_jacobian = F;
            // step_V = V;
            jacobian = F * jacobian;
            covariance_ = F * covariance_ * F.transpose() + V * noise_ * V.transpose();
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief 
     * 
     * @param dt 
     * @param acc_1 
     * @param gyr_1 
     */
    void propagate(const double& dt, const Eigen::Vector3d &acc_1, const Eigen::Vector3d &gyr_1) {
        Eigen::Vector3d result_delta_p;
        Eigen::Quaterniond result_delta_q;
        Eigen::Vector3d result_delta_v;
        Eigen::Vector3d result_linearized_ba;
        Eigen::Vector3d result_linearized_bg;

        midPointIntegration(dt, acc_0_, gyr_0_, acc_1, gyr_1, delta_p_, delta_q_, delta_v_,
                            linearized_ba_, linearized_bg_,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, 1);

        //checkJacobian(dt, acc_0_, gyr_0_, acc_1_, gyr_1_, delta_p, delta_q, delta_v,
        //                    linearized_ba_, linearized_bg_);
        delta_p_ = result_delta_p;
        delta_q_ = result_delta_q;
        delta_v_ = result_delta_v;
        linearized_ba_ = result_linearized_ba;
        linearized_bg_ = result_linearized_bg;
        delta_q_.normalize();
        sum_dt_ += dt;
        acc_0_ = acc_1;
        gyr_0_ = gyr_1;  
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief 重新更新残差 
     * 
     * @param Pi 
     * @param Qi 
     * @param Vi 
     * @param Bai 
     * @param Bgi 
     * @param Pj 
     * @param Qj 
     * @param Vj 
     * @param Baj 
     * @param Bgj 
     * @return Eigen::Matrix<double, 15, 1> 
     */
    Eigen::Matrix<double, 15, 1> evaluate(
            const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, 
            const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
            const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, 
            const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj) {
        Eigen::Matrix<double, 15, 1> residuals;

        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        Eigen::Vector3d dba = Bai - linearized_ba_;
        Eigen::Vector3d dbg = Bgi - linearized_bg_;

        Eigen::Quaterniond corrected_delta_q = delta_q_ * Math::deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_delta_v = delta_v_ + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_delta_p = delta_p_ + dp_dba * dba + dp_dbg * dbg;

        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * G * sum_dt_ * sum_dt_ + Pj - Pi - Vi * sum_dt_) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt_ + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }

    const Eigen::Vector3d linearized_acc, linearized_gyr;
    Eigen::Vector3d linearized_ba_, linearized_bg_;

    Eigen::Matrix<double, 15, 15> jacobian, covariance_;
    Eigen::Matrix<double, 18, 18> noise_;

    double sum_dt_;
    Eigen::Vector3d delta_p_;
    Eigen::Quaterniond delta_q_;
    Eigen::Vector3d delta_v_;

    std::vector<double> dt_buf;
    std::vector<Eigen::Vector3d> acc_buf;
    std::vector<Eigen::Vector3d> gyr_buf;
private:
    double ACC_N_, GYR_N_;     // 测量噪声
    double ACC_W_, GYR_W_;    // 随机游走噪声
    double curr_time_ = -1;
    Eigen::Vector3d acc_0_{-1, -1, -1}, gyr_0_{-1, -1, -1};
};
}
}
