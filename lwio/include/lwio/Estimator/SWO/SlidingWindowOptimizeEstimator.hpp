
#pragma once 
#include <ceres/ceres.h>
#include "../ceres/pose_local_parameterization.hpp"
#include "../preintegration/imu_preintegration_base.hpp"
#include "../ceres/imu_preIntegration_factor.hpp"
#include "../ceres/odom_factor.hpp"

namespace lwio {
namespace estimator {

/**
 * @brief 基于滑动窗口的估计器
 * 
 */
class SlidingWindowOptimazerEstimate {
public:
    struct Option {
        //  IMU 参数配置  
        double IMU_ACC_N_, IMU_GYR_N_;     // 测量噪声
        double IMU_ACC_W_, IMU_GYR_W_;    // 随机游走噪声
    };

    SlidingWindowOptimazerEstimate() {}
    SlidingWindowOptimazerEstimate(Option option) : option_(option) {
        clearState();  
    } 

    /**
     * @brief 对估计器进行初始化
     * @details 设置滑窗第一帧  
     * 
     */
    void Initialize() {
        ++frame_count_;    // 初始化后由0变为1了 
    }

    /**
     * @brief 滑动窗口估计
     * 
     * @param imu_data 当前激光雷达帧对应的IMU数据
     * @param motion 激光雷达里程计的运动
     * @return true 
     * @return false 
     */
    bool Estimate(const std::deque<sensor::ImuData>& imu_data, 
                                    const Eigen::Isometry3d& motion, const Eigen::Matrix<double, 6, 6>& lidarOdom_cov) {
        if (frame_count_ == 0)  return false;   
        updateCeresParam(); 
        // 进行预积分
        imuPreIntegrations(imu_data);     // imu预积分  
        // 设置最新激光帧的P，V状态
        last_pose_ = last_pose_ * motion;  
        addCeresPoseParam(last_pose_);
        // frame_count_ 表示当前滑动窗口内所拥有的帧数量  
        if (frame_count_ < Param::WINDOW_SIZE_) {
            ++frame_count_;
        }

        ceres::Problem problem;
        ceres::LossFunction *loss_function;
        //loss_function = new ceres::HuberLoss(1.0);
        loss_function = new ceres::CauchyLoss(1.0);
        // 加入滑动窗口内的所有机器人状态 
        for (int i = 0; i < frame_count_; i++) {
            ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
            problem.AddParameterBlock(ceres_param_pose_[i], Param::SIZE_POSE_, local_parameterization);
            problem.AddParameterBlock(ceres_param_SpeedBias_[i], Param::SIZE_SPEEDBIAS_);
        }
        // 滑动窗口满了以后，将第一帧给fix住
        /**
         * @todo 这里是简单的fix住，以后可以尝试边缘化 + fej  
         * 
         */
        if (frame_count_ == Param::WINDOW_SIZE_) {
            problem.SetParameterBlockConstant(ceres_param_pose_[0]);
            problem.SetParameterBlockConstant(ceres_param_SpeedBias_[0]);
        }

        // 加入外参状态——激光雷达相对与IMU
        // 如果估计时间戳对齐，这里要加入相关状态
        // 加入边缘化先验残差约束 —— 激光雷达融合系统不需要，融合视觉后需要加入
        // 加入IMU 预积分、激光里程计、轮速里程计的残差约束
        for (int i = 0; i < Param::WINDOW_SIZE_ - 1; i++) {
            // 添加IMU预积分因子  
            if (imu_pre_integrations_[i]->sum_dt_ > 10.0)
                continue;
            IMUPreIntegrationFactor* imu_factor = new IMUPreIntegrationFactor(imu_pre_integrations_[i]);
            problem.AddResidualBlock(imu_factor, NULL, ceres_param_pose_[i], ceres_param_SpeedBias_[i], ceres_param_pose_[i + 1], ceres_param_SpeedBias_[i + 1]);
            // 添加激光里程计因子
            OdomFactor* lidar_odom_factor = new OdomFactor(motion, lidarOdom_cov); 
            problem.AddResidualBlock(lidar_odom_factor, NULL, ceres_param_pose_[i], ceres_param_pose_[i + 1]); 
        }
        // 进行优化
        ceres::Solver::Options options;

        options.linear_solver_type = ceres::DENSE_SCHUR;
        //options.num_threads = 2;
        options.trust_region_strategy_type = ceres::DOGLEG;
        // options.max_num_iterations = NUM_ITERATIONS;
        //options.use_explicit_schur_complement = true;
        //options.minimizer_progress_to_stdout = true;
        //options.use_nonmonotonic_steps = true;
        // TicToc t_solver;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        updateStateFromCeres();  
        // 进行边缘化 
        // 融合了视觉就需要边缘化，如果是激光雷达系统，直接将状态和约束丢弃即可
        slidingWindow();  
    }

private:
    void clearState() {
        for (int i = 0; i < Param::WINDOW_SIZE_; i++) {
            Rs_[i].setIdentity();
            Ps_[i].setZero();
            Vs_[i].setZero();
            Bas_[i].setZero();
            Bgs_[i].setZero();
            dt_buf_[i].clear();
            linear_acceleration_buf_[i].clear();
            angular_velocity_buf_[i].clear();

            if (imu_pre_integrations_[i] != nullptr)
                delete imu_pre_integrations_[i];
            imu_pre_integrations_[i] = nullptr;
        }
    }

    /**
     * @brief ceres 优化后 ，用ceres优化后的内部状态更新当前状态
     * 
     */
    void updateStateFromCeres() {
        // 由与yaw不可观，因此，如果yaw被错误的优化了，在这里恢复  
        Eigen::Vector3d origin_R0 = Math::R2ypr(Rs_[0]);   // 优化前滑窗第一个关键帧node的旋转，要与优化后的结果进行比较 
        // Eigen::Vector3d origin_P0 = Ps[0];
        // 优化后滑窗第一帧的旋转  
        Eigen::Vector3d origin_R0_after_opt = Math::R2ypr(Eigen::Quaterniond(ceres_param_pose_[0][6],
                                                        ceres_param_pose_[0][3],
                                                        ceres_param_pose_[0][4],
                                                        ceres_param_pose_[0][5]).toRotationMatrix());
        // 计算优化前后yaw的变化量
        double y_diff = origin_R0.x() - origin_R0_after_opt.x();
        // TODO
        Eigen::Matrix3d rot_diff = Math::ypr2R(Eigen::Vector3d(y_diff, 0, 0));
        if (std::abs(std::abs(origin_R0.y()) - 90) < 1.0 || std::abs(std::abs(origin_R0_after_opt.y()) - 90) < 1.0) {
            rot_diff = Rs_[0] * Eigen::Quaterniond(ceres_param_pose_[0][6],
                                        ceres_param_pose_[0][3],
                                        ceres_param_pose_[0][4],
                                        ceres_param_pose_[0][5]).toRotationMatrix().transpose();
        }

        for (int i = 0; i < frame_count_; i++) {
            // 如果yaw被错误的改变了，那么用rot_diff 将yaw校正回去   
            Rs_[i] = rot_diff * Eigen::Quaterniond(ceres_param_pose_[i][6], 
                                                                                        ceres_param_pose_[i][3], 
                                                                                        ceres_param_pose_[i][4], 
                                                                                        ceres_param_pose_[i][5]).normalized().toRotationMatrix();

            // Ps_[i] = rot_diff * Eigen::Vector3d(ceres_param_pose_[i][0] - ceres_param_pose_[0][0],
            //                         ceres_param_pose_[i][1] - ceres_param_pose_[0][1],
            //                         ceres_param_pose_[i][2] - ceres_param_pose_[0][2]) + origin_P0;
            // 感觉原式子有误(如上)，不应该是 + origin_P0
            Ps_[i] = rot_diff * Eigen::Vector3d(ceres_param_pose_[i][0] - ceres_param_pose_[0][0],
                        ceres_param_pose_[i][1] - ceres_param_pose_[0][1],
                        ceres_param_pose_[i][2] - ceres_param_pose_[0][2]) 
                        + Eigen::Vector3d(ceres_param_pose_[0][0], ceres_param_pose_[0][1], ceres_param_pose_[0][2]);

            Vs_[i] = rot_diff * Eigen::Vector3d(ceres_param_SpeedBias_[i][0],
                                        ceres_param_SpeedBias_[i][1],
                                        ceres_param_SpeedBias_[i][2]);

            Bas_[i] = Eigen::Vector3d(ceres_param_SpeedBias_[i][3],
                            ceres_param_SpeedBias_[i][4],
                            ceres_param_SpeedBias_[i][5]);

            Bgs_[i] = Eigen::Vector3d(ceres_param_SpeedBias_[i][6],
                            ceres_param_SpeedBias_[i][7],
                            ceres_param_SpeedBias_[i][8]);
        }
    }

    /**
     * @brief 将当前滑窗内的状态   更新到ceres的优化状态参数中，准备优化 
     * 
     */
    void updateCeresParam() {
        for (int i = 0; i < frame_count_; i++) {
            ceres_param_pose_[i][0] = Ps_[i].x();
            ceres_param_pose_[i][1] = Ps_[i].y();
            ceres_param_pose_[i][2] = Ps_[i].z();
            Eigen::Quaterniond q{Rs_[i]};
            ceres_param_pose_[i][3] = q.x();
            ceres_param_pose_[i][4] = q.y();
            ceres_param_pose_[i][5] = q.z();
            ceres_param_pose_[i][6] = q.w();

            ceres_param_SpeedBias_[i][0] = Vs_[i].x();
            ceres_param_SpeedBias_[i][1] = Vs_[i].y();
            ceres_param_SpeedBias_[i][2] = Vs_[i].z();

            ceres_param_SpeedBias_[i][3] = Bas_[i].x();
            ceres_param_SpeedBias_[i][4] = Bas_[i].y();
            ceres_param_SpeedBias_[i][5] = Bas_[i].z();

            ceres_param_SpeedBias_[i][6] = Bgs_[i].x();
            ceres_param_SpeedBias_[i][7] = Bgs_[i].y();
            ceres_param_SpeedBias_[i][8] = Bgs_[i].z();
        }
    }

    /**
     * @brief 
     * 
     * @param imu_data 
     */
    void imuPreIntegrations(const std::deque<sensor::ImuData>& imu_data) {
        if (imu_pre_integrations_[frame_count_ - 1] == nullptr) {
            imu_pre_integrations_[frame_count_ - 1] = 
                new ImuPreIntegrationBase{option_.IMU_ACC_N_, option_.IMU_GYR_N_,
                    option_.IMU_ACC_W_, option_.IMU_GYR_W_,
                    Bas_[frame_count_ - 1], Bgs_[frame_count_ - 1]};   // bias用
        }
        for (const auto& imu : imu_data) {
            imu_pre_integrations_[frame_count_ - 1]->push_back(imu.timestamp_, imu.acc_, imu.gyro_);
        }
    }

    void addCeresPoseParam(const Eigen::Isometry3d& pose) {
        const auto& t = pose.translation();
        ceres_param_pose_[frame_count_][0] = t.x();
        ceres_param_pose_[frame_count_][1] = t.y();
        ceres_param_pose_[frame_count_][2] = t.z();
        Eigen::Quaterniond q{pose.linear()};
        ceres_param_pose_[frame_count_][3] = q.x();
        ceres_param_pose_[frame_count_][4] = q.y();
        ceres_param_pose_[frame_count_][5] = q.z();
        ceres_param_pose_[frame_count_][6] = q.w();
    }

    void slidingWindow() {

    }

    struct Param {
        static const int WINDOW_SIZE_ = 20;
        static const int SIZE_POSE_ = 7;
        static const int SIZE_SPEEDBIAS_ = 9;
    };

    Option option_;
    int frame_count_ = 0;     // 当前滑动窗口中，帧的数量  

    double ceres_param_pose_[Param::WINDOW_SIZE_][Param::SIZE_POSE_];
    double ceres_param_SpeedBias_[Param::WINDOW_SIZE_][Param::SIZE_SPEEDBIAS_];
    // 系统状态 
    Eigen::Vector3d Ps_[Param::WINDOW_SIZE_];
    Eigen::Vector3d Vs_[Param::WINDOW_SIZE_];
    Eigen::Matrix3d Rs_[Param::WINDOW_SIZE_];
    Eigen::Vector3d Bas_[Param::WINDOW_SIZE_];
    Eigen::Vector3d Bgs_[Param::WINDOW_SIZE_];
    ImuPreIntegrationBase *imu_pre_integrations_[Param::WINDOW_SIZE_];
    std::vector<double> dt_buf_[Param::WINDOW_SIZE_];
    std::vector<Eigen::Vector3d> linear_acceleration_buf_[Param::WINDOW_SIZE_];
    std::vector<Eigen::Vector3d> angular_velocity_buf_[Param::WINDOW_SIZE_];

    Eigen::Isometry3d last_pose_ = Eigen::Isometry3d::Identity();  
};

}
}