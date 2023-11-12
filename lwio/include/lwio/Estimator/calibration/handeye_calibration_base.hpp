#pragma once 

#include <queue>
#include <eigen3/Eigen/Dense>
#include "lwio/Common/pose.hpp"
#include "lwio/Math.hpp"
#include "lwio/Common/color.hpp"

namespace lwio {

// 用于优先队列 将旋转小的元素放到前面 
// 大顶堆   w 越大  theta 越小  
struct rotCmp {
    bool operator()(const std::pair<uint16_t, std::pair<Pose, Pose>> &pose_r, 
                                    const std::pair<uint16_t, std::pair<Pose, Pose>> &pose_l) {   
        // 大顶堆 
        return (pose_l.second.first.q().w() > pose_r.second.first.q().w());    // w = cos(theta/2)    
    }
};

/**
 * @brief: 手眼标定的实现基础实现
 * @details: 
 */    
class HandEyeCalibrationBase {
    public:  
        HandEyeCalibrationBase() {
            Q_ = Eigen::MatrixXd::Zero(N_POSE_ * 4, 4);
            pose_storage_.reserve(N_POSE_);  
            rot_cov_thre_ = 0.22;  
            ext_q_result_ = Eigen::Quaterniond(0.0, 0.0, 0.0, 0.0);
            ext_t_result_.setZero();  
            primary_lidar_accum_pose_.SetIdentity();
            sub_lidar_accum_pose_.SetIdentity();  
        }

        /**
         * @brief 添加一组pose数据    
         * @param pose_primary 主传感器的pose
         * @param pose_sub 辅传感器pose 
         * @param trans_valid pose中平移是否有效  
         * @brief 添加一对传感器的运动数据  
         * @return 本组运动是否合格   
         */
        bool AddPose(const  Pose& pose_primary, const  Pose& pose_sub, 
                const bool& trans_valid = true) {        
            // 首先检查pose 
            if (!checkScrewMotion(pose_primary, pose_sub, trans_valid)) {
                // std::cout<<"ScrewMotion error"<<std::endl;
                return false;
            }

            if (pose_storage_.size() < N_POSE_) {
                new_pose_pair_.emplace(pose_storage_.size(), 
                    std::make_pair(primary_lidar_accum_pose_, sub_lidar_accum_pose_));
                priority_pose_.emplace(pose_storage_.size(), 
                    std::make_pair(primary_lidar_accum_pose_, sub_lidar_accum_pose_));  
                pose_storage_.emplace_back(primary_lidar_accum_pose_, sub_lidar_accum_pose_); 
            }  else {  
                // 数据量大于设定值    则滑动窗口 
                uint16_t pos = priority_pose_.top().first;     // 旋转最小的运动的序号   
                // std::cout<<"num > 300, remove top w: "<< priority_pose_.top().second.first.q().w()<<std::endl;
                pose_storage_[pos] = std::make_pair(primary_lidar_accum_pose_, sub_lidar_accum_pose_);  
                new_pose_pair_.emplace(pos, std::make_pair(primary_lidar_accum_pose_, sub_lidar_accum_pose_));
                priority_pose_.pop();   
                priority_pose_.emplace(pos, std::make_pair(primary_lidar_accum_pose_, sub_lidar_accum_pose_));
            }   

            primary_lidar_accum_pose_.SetIdentity();
            sub_lidar_accum_pose_.SetIdentity();  
            std::cout<<"pose_storage_.size(): "<< pose_storage_.size()<<std::endl;
            if (pose_storage_.size() >= 10) return true;    // 至少有10个数据才认为可以求解  
            return false;  
        }

        /**
         * @brief: 标定旋转 
         * @param[out] ext_q 标定出的旋转外参  
         * @return 是否成功
         */            
        bool CalibExRotation() {
            while (!new_pose_pair_.empty()) {
                auto pose_pair = new_pose_pair_.front();  
                new_pose_pair_.pop();
                
                Eigen::Quaterniond const& primary_q = pose_pair.second.first.q();    // 主传感器的旋转
                Eigen::Quaterniond const& sub_q = pose_pair.second.second.q();  // 辅传感器的旋转  
                uint16_t const& indice = pose_pair.first;  
                // 求rubust核函数
                // 当求解出初步的外参后，通过这一步削弱明显错误的运动数据的影响  
                double huber = 1.0;   // the derivative of huber norm
                if (ext_q_result_.w() != 0) {
                    Eigen::Quaterniond ref_q = ext_q_result_ * sub_q * ext_q_result_.inverse();
                    double angular_distance = 180 / M_PI * primary_q.angularDistance(ref_q); // calculate delta_theta=|theta_1-theta_2|
                    huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
                }
                Eigen::Matrix4d L = Math::QLeft(primary_q);  
                Eigen::Matrix4d R = Math::QRight(sub_q);  
                Q_.block<4, 4>(indice * 4, 0) = huber * (L - R);
            }
            // SVD 求解 AX = 0 ，|X| = 1  
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(Q_, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix<double, 4, 1> x = svd.matrixV().col(3); // [w, x, y, z]     // 最小奇异值对应的特征向量为解
            if (x[0] < 0) x = -x; // use the standard quaternion
            Eigen::Vector4d rot_cov = svd.singularValues();    // singular value
            std::cout<<"rot_cov: "<<rot_cov.transpose()<<std::endl;
            // 若没有退化  AQ = 0 零空间为1 ， 则A的秩 = 4 -1 = 3， 那么ATA秩也为3， 因此A的奇异值只有一个为0
            //  因此 检查第二小奇异值， 看是否足够大，越大，解越稳定
            if (rot_cov[2] > rot_cov_thre_) {
                ext_q_result_ = Eigen::Quaterniond(x[0], x[1], x[2], x[3]);
                ext_q_result_.normalize(); 
                std::cout << "rot calibration done, ext_q_result_ w: " << ext_q_result_.w() << ",vec: " 
                    << ext_q_result_.vec().transpose() << std::endl;
                return true; 
            } else {
                std::cout << SlamLib::color::RED << "calibration error ! 第二小奇异值: " << rot_cov[2] 
                    << SlamLib::color::RESET << std::endl; 
            }
            return false;  
        }

        bool CalibExTranslation() {
            if (calibExTranslationNonPlanar()) {
                calib_done_ = true;
                return true;
            }
            return false;  
        }

        bool calibExTranslationNonPlanar() {
            Eigen::MatrixXd A = Eigen::MatrixXd::Zero(pose_storage_.size() * 3, 3);
            Eigen::MatrixXd b = Eigen::MatrixXd::Zero(pose_storage_.size() * 3, 1);
            for (size_t i = 0; i < pose_storage_.size(); i++) {
                const Pose &pose_primary = pose_storage_[i].first;
                const Pose &pose_sub = pose_storage_[i].second;
                // AngleAxisd ang_axis_ref(pose_primary.q_);
                // AngleAxisd ang_axis_data(pose_sub.q_);
                // // 计算指向旋转轴方向的平移差值 
                // double t_dis = abs(pose_primary.t_.dot(ang_axis_ref.axis()) - pose_sub.t_.dot(ang_axis_data.axis()));
                // double huber = t_dis > 0.04 ? 0.04 / t_dis : 1.0;
                A.block<3, 3>(i * 3, 0) = (pose_primary.q().toRotationMatrix() - Eigen::Matrix3d::Identity());
                b.block<3, 1>(i * 3, 0) = ext_q_result_ * pose_sub.t() - pose_primary.t();
            }
            Eigen::Vector3d x; 
            /**
             *  TODO: 看看和QR分解的差别 
             */
            x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
            ext_t_result_ = x;

            return true;
        }

        /**
         * @brief 平面运动时的旋转估计  
         * 
         * @return true 
         * @return false 
         */
        bool CalibExRotationPlanar() {
            while (!new_pose_pair_.empty()) {
                auto pose_pair = new_pose_pair_.front();  
                new_pose_pair_.pop();
                
                Eigen::Quaterniond primary_q = pose_pair.second.first.q();    // 主传感器的旋转
                Eigen::Quaterniond const& sub_q = pose_pair.second.second.q();  // 辅传感器的旋转  
                std::cout << "sub_q: " << sub_q.w() << ", " << sub_q.vec().transpose() << std::endl;
                std::cout << "primary_q: " << primary_q.w() << ", " << primary_q.vec().transpose() << std::endl;
                // 将主传感器的旋转调整为与z轴平行
                primary_q.x() = 0; 
                primary_q.y() = 0; 
                if (primary_q.z() < 0) {
                    primary_q.z() = -std::sqrt(1 - primary_q.w());
                } else {
                    primary_q.z() = std::sqrt(1 - primary_q.w());
                }
                std::cout << "modified primary_q: " << primary_q.w() << ", " << primary_q.vec().transpose() << std::endl;
                uint16_t const& indice = pose_pair.first;  
                // 求rubust核函数
                // 当求解出初步的外参后，通过这一步削弱明显错误的运动数据的影响  
                double huber = 1.0;   // the derivative of huber norm
                // if (ext_q_result_.w() != 0) {
                //     Eigen::Quaterniond ref_q = ext_q_result_ * sub_q * ext_q_result_.inverse();
                //     double angular_distance = 180 / M_PI * primary_q.angularDistance(ref_q); // calculate delta_theta=|theta_1-theta_2|
                //     huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
                // }
                Eigen::Matrix4d L = Math::QLeft(primary_q);  
                Eigen::Matrix4d R = Math::QRight(sub_q);  
                Q_.block<4, 4>(indice * 4, 0) = huber * (L - R);
            }
            // SVD 求解 AX = 0 ，|X| = 1 ， 这里 X表示 sub->primary的旋转
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(Q_, Eigen::ComputeFullU | Eigen::ComputeFullV);
            // 检查最小的两个特征值是否足够的小
            Eigen::Vector4d singular_values = svd.singularValues();    // singular value
            std::cout<<"singular_values: "<<singular_values.transpose()<<std::endl;
            if (singular_values[1] <= rot_cov_thre_ || singular_values[2] >= rot_cov_thre_) {
                std::cout << SlamLib::color::RED << "CalibExRotationPlanar() error ! 奇异值有误 "
                    << SlamLib::color::RESET << std::endl; 
                return false;  
            }
            // 提取出最小的两个特征值对应的特征向量
            Eigen::Matrix<double, 4, 1> t_1 = svd.matrixV().col(3); // [w, x, y, z]   
            Eigen::Matrix<double, 4, 1> t_2 = svd.matrixV().col(2); // [w, x, y, z]   
                         
            // 计算 q_yz  , q_yz = a * t_1 + b * t_2
            // 先考虑约束  q_yz(w)*q_yz(x) = q_wz(y) * q_wz(z)
            // 求解一元二次方程求解 a / b
            double X[2];
            if (!Math::solveQuadraticEquation<double>(
                    t_1[0] * t_1[1] - t_1[2] * t_1[3], 
                    t_1[0] * t_2[1] + t_2[0] * t_1[1] - t_1[2] * t_2[3] - t_1[3] * t_2[2], 
                    t_2[0] * t_2[1] - t_2[2] * t_2[3], 
                    X[0], X[1])) {
                std::cout << SlamLib::color::RED << "求解二次方程计算a / b 无解！" << SlamLib::color::RESET << std::endl; 
                return false;
            }
            std::cout << SlamLib::color::GREEN << "解得 X[0]: " <<  X[0] << ", X[1]: " << X[0] << std::endl;
            // 考虑 |q_yz| = 1 约束
            double b = std::sqrt(1.0 / (X[0] * X[0] * t_1.dot(t_1) + t_2.dot(t_2) + 2 * X[0] * t_1.dot(t_2)));
            Eigen::Matrix<double, 4, 1> q_yz = X[0] * b * t_1 + b * t_2;  
            if (q_yz[0] < 0) q_yz = -q_yz;  
            // 验证
            std::cout << SlamLib::color::GREEN << "|q_yz|: " << q_yz.norm() << std::endl;
            std::cout << "q_yz1 * q_yz2: " << q_yz[0] * q_yz[1] << ", q_yz3 * q_yz4:" << q_yz[2] * q_yz[3] 
                << SlamLib::color::RESET << std::endl; 
            std::cout << "q_yz: " << q_yz.transpose() << std::endl;
            ext_q_result_ = Eigen::Quaterniond(q_yz[0], q_yz[1], q_yz[2], q_yz[3]);
            return true;  
            // // 若没有退化  AQ = 0 零空间为1 ， 则A的秩 = 4 -1 = 3， 那么ATA秩也为3， 因此A的奇异值只有一个为0
            // //  因此 检查第二小奇异值， 看是否足够大，越大，解越稳定
            // if (rot_cov[2] > rot_cov_thre_) {
            //     ext_q_result_ = Eigen::Quaterniond(x[0], x[1], x[2], x[3]);
            //     ext_q_result_.normalize(); 
            //     std::cout << "rot calibration done, ext_q_result_ w: " << ext_q_result_.w() << ",vec: " 
            //         << ext_q_result_.vec().transpose() << std::endl;
            //     return true; 
            // } else {
            //     std::cout << SlamLib::color::RED << "calibration error ! 第二小奇异值: " << rot_cov[2] 
            //         << SlamLib::color::RESET << std::endl; 
            // }
            // return false;  
        }

        bool CalibExTranslationPlanar() {

            return false;  
        }

        void GetCalibResult(Eigen::Isometry3d& result) {
            result.linear() = ext_q_result_.toRotationMatrix();
            result.translation() = ext_t_result_;  
        }

        const Eigen::Quaterniond& GetCalibRot() const {
            return ext_q_result_;
        }

        const Eigen::Vector3d& GetCalibTrans() const {
            return ext_t_result_;
        }

    protected:
        /**
         * @brief: 检查运动是否符合条件 
         * @details: 
         * @param {Isometry3d} &pose_primary
         * @param {Isometry3d} &pose_sub
         * @return {*}
         */            
        bool checkScrewMotion(const Pose &pose_primary, const Pose &pose_sub, const bool& trans_valid) {
            Eigen::AngleAxisd ang_axis_pri(pose_primary.q());
            Eigen::AngleAxisd ang_axis_sub(pose_sub.q());
            //  检查当前数据是否正确   刚性连接的传感器   旋转和平移应该满足下面的关系 
            //  检查旋转  
            double r_dis = abs(ang_axis_pri.angle() - ang_axis_sub.angle());   
            if (r_dis > EPSILON_R_) {
                std::cout << SlamLib::color::RED <<
                    "HandEyeCalibrationBase::checkScrewMotion -- r_dis > EPSILON_R_ !!"
                    << SlamLib::color::RESET << std::endl;
                primary_lidar_accum_pose_.SetIdentity();
                sub_lidar_accum_pose_.SetIdentity();  
                return false;
            }
            // 检查平移 
            if (trans_valid) {    // 首先平移数据要有效  
                double t_dis = abs(pose_primary.t().dot(ang_axis_pri.axis()) 
                                                        - pose_sub.t().dot(ang_axis_sub.axis()));
                if (t_dis > EPSILON_T_) {
                    // std::cout<<common::RED<<"t_dis > EPSILON_T_!!"<<std::endl;
                    primary_lidar_accum_pose_.SetIdentity();
                    sub_lidar_accum_pose_.SetIdentity();  
                    return false;
                }
            }
            // 如果旋转不够的话   则进行累计直到旋转足够  
            primary_lidar_accum_pose_ = primary_lidar_accum_pose_ * pose_primary;
            sub_lidar_accum_pose_ = sub_lidar_accum_pose_ * pose_sub; 
            Eigen::AngleAxisd ang_axis_accum_pri(primary_lidar_accum_pose_.q());
            Eigen::AngleAxisd ang_axis_accum_sub(sub_lidar_accum_pose_.q());

            if (ang_axis_accum_pri.angle() > 0.035 || ang_axis_accum_sub.angle() > 0.035) {
                return true;  
            }
            return false;  
        }

    private:
        const double EPSILON_R_ = 0.05;
        const double EPSILON_T_ = 0.1;
        const size_t N_POSE_ = 300;              // 最多包含的pose个数   
        double rot_cov_thre_;
        // 对pose 按照质量进行排序  
        std::priority_queue<std::pair<uint16_t, std::pair<Pose, Pose>>, 
            std::vector<std::pair<uint16_t, std::pair<Pose, Pose>>>, rotCmp> priority_pose_;
        // pose 数据库中的序号  
        std::queue<std::pair<uint16_t, std::pair<Pose, Pose>>> new_pose_pair_;    
        std::vector<std::pair<Pose, Pose>> pose_storage_;   

        Eigen::MatrixXd Q_;    // 计算旋转外参时的矩阵  
        Eigen::Quaterniond ext_q_result_;  
        Eigen::Vector3d ext_t_result_;  
        bool calib_done_ = false; 
        Pose primary_lidar_accum_pose_, sub_lidar_accum_pose_;
};
}