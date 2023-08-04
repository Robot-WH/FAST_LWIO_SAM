/*
 * @Copyright(C): 
 * @Description: Lidar-IMU-GNSS 的基于GTSAM 因子图的前端估计器
 */
#pragma once 
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include "LIGEstimator_interface.hpp"
#include "LidarTracker/LidarTracker.hpp"

namespace Slam3D {
namespace Estimator {
namespace {

using namespace gtsam;

using symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)
using symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::G; // GPS pose

/**
 * @brief: lidar-imu-gnss 因子图估计器 
 * @details:  融合IMU与GNSS，有IMU时融合IMU，没有IMU数据时，采用 运动学模型  
 * @param _PointT 估计器处理的激光点类型 
 * @param _StateType 估计状态的类型  
 */    
template<typename _PointT, typename _StateType>
class LIGFactorGraphEstimator : public LIGEstimatorInterface<_PointT> {
    public:
        struct Option {
            // 激光IMU外参
            Eigen::Isometry3d ext_imu_lidar_T;     // lidar->imu
            float imuAccNoise_;          // 加速度噪声标准差
            float imuGyrNoise_;          // 角速度噪声标准差
            float imuAccBiasN_;          // bias 随机游走噪声  
            float imuGyrBiasN_;
            float imuGravity_;     // 重力加速度
        };

        LIGFactorGraphEstimator() {
            // imu预积分的噪声协方差
            boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(option_.imuGravity_);
            p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(option_.imuAccNoise_, 2); // acc white noise in continuous
            p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(option_.imuGyrNoise_, 2); // gyro white noise in continuous
            p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
            gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias
            // 噪声先验
            priorPoseNoise_  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
            priorVelNoise_   = gtsam::noiseModel::Isotropic::Sigma(3, 1e2); // m/s
            priorBiasNoise_  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
            // 激光里程计scan-to-map优化过程中发生退化，则选择一个较大的协方差
            correctionNoise_ = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
            correctionDegenerateNoise_ = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // rad,rad,rad,m, m, m
            noiseModelBetweenBias_ = (gtsam::Vector(6) << option_.imuAccBiasN_, option_.imuAccBiasN_, option_.imuAccBiasN_, 
                option_.imuGyrBiasN_, option_.imuGyrBiasN_, option_.imuGyrBiasN_).finished();
            // imu预积分器，用于因子图优化
            imuIntegrator_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization        
        }

        virtual bool Initialize() override {
        }

        virtual void ProcessData(const LidarImuDataPacket<_PointT> &data) override {
            TicToc tt;
            // 获取预测位姿
            Eigen::Isometry3d motion = Eigen::Isometry3d::Identity();
            double time_begin_ = data.feature_point_.timestamp_start_;
            double time_end_ = 0.;
            lidar_trackers_->Solve(data.feature_point_, motion);
            tt.toc("lidar tracker "); 
            // IMU预积分

        }

        _StateType GetCurrState() {
            return state_;  
        }

    private:
        Option option_;  
        std::unique_ptr<LidarTrackerBase<_PointT>> lidar_trackers_;  
        _StateType state_;  // 估计的状态  

        // 噪声协方差
        gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise_;
        gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise_;
        gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise_;
        gtsam::noiseModel::Diagonal::shared_ptr correctionNoise_;
        gtsam::noiseModel::Diagonal::shared_ptr correctionDegenerateNoise_;
        gtsam::Vector noiseModelBetweenBias_;

        // imu预积分器
        gtsam::PreintegratedImuMeasurements *imuIntegrator_;

        // imu因子图优化过程中的状态变量
        gtsam::Pose3 prevPose_;
        gtsam::Vector3 prevVel_;
        gtsam::NavState prevState_;
        gtsam::imuBias::ConstantBias prevBias_;

        // imu-lidar位姿变换
        gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), 
            gtsam::Point3(-option_.ext_imu_lidar_T.translation().x(), 
                                            -option_.ext_imu_lidar_T.translation().y(), 
                                            -option_.ext_imu_lidar_T.translation().z()));
        gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), 
            gtsam::Point3(option_.ext_imu_lidar_T.translation().x(), 
                                            option_.ext_imu_lidar_T.translation().y(), 
                                            option_.ext_imu_lidar_T.translation().z()));
        // ISAM2优化器
        gtsam::ISAM2 optimizer;
        gtsam::NonlinearFactorGraph graphFactors;
        gtsam::Values graphValues;
};
}
}
}
