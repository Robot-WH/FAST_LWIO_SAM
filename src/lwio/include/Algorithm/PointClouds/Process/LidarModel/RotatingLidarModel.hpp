
/*
 * @Copyright(C): 
 * @FileName: 文件名
 * @Author: lwh
 */
#pragma once 

#include "Sensor/lidar_data_type.h"
#include "Common/pcl_type.h"

namespace Slam3D {
namespace Algorithm {
    
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: 机械旋转雷达模型    常用 16、32、64线 雷达
 *                  分线束模型
 */    
template<typename _PointT>
class RotatingLidarModel {
    public:
        struct Option {
            uint16_t line_num_;    // 线数
            float horizon_angle_resolution_;  // 水平分辨率
        };
        RotatingLidarModel() {}
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        RotatingLidarModel(Option const& option) 
        : option_(option) {
            init();
        }
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 激光内参自标定
         * @details: 
         */        
        bool LidarIntrinsicCalibrate(pcl::PointCloud<_PointT> const& pointcloud) {
            std::set<float> scan_angle_set; 
            std::map<float, std::set<float>> scan_horizon_angle; 
            float error_sigma = 0.2;
            // step1: 求解参数 - 1、有多少线，每个线的角度    2、将每个点按照角度排序放到各自的scan中
            for (uint32_t i = 0; i < pointcloud.points.size(); i++) {
                float horizon_angle = atan2(pointcloud.points[i].x, pointcloud.points[i].y) * 180 / M_PI;    // atan2(y, x) 
                float vertical_angle = atan(pointcloud.points[i].z / sqrt(pointcloud.points[i].x * pointcloud.points[i].x 
                    + pointcloud.points[i].y * pointcloud.points[i].y)) * 180 / M_PI;
                // 设定每次求解误差直径 < r(误差分布长度) ,线束之间的最小分辨距离 > 2 * r
                if (scan_angle_set.empty()) {
                    scan_angle_set.insert(vertical_angle); 
                    scan_horizon_angle[vertical_angle].insert(horizon_angle); 
                } else {
                    auto it = scan_angle_set.lower_bound(vertical_angle);   // 二分查找  返回>= , *scan_angle_set.end() == 0
                    if (it == scan_angle_set.end()) {
                        it--;  
                    }
                    // 误差半径 < 0.1 ,一个scan的最大范围即0.1*2     超过这个范围  就认为不是一个scan 
                    if (fabs(*it - vertical_angle) >= 2 * error_sigma) {
                        if (*it < vertical_angle) {
                            scan_angle_set.insert(vertical_angle);
                            scan_horizon_angle[vertical_angle].insert(horizon_angle); 
                            //std::cout<<std::setprecision(16)<<"scan_horizon_angle insert:"<<vertical_angle<<std::endl;
                        } else {
                            it = scan_angle_set.lower_bound(vertical_angle - 2 * error_sigma); 
                            if (fabs(*it - vertical_angle) >= 2 * error_sigma) {
                                scan_angle_set.insert(vertical_angle);
                                scan_horizon_angle[vertical_angle].insert(horizon_angle); 
                                //std::cout<<std::setprecision(16)<<"scan_horizon_angle insert:"<<vertical_angle<<std::endl;
                            } else {
                                scan_horizon_angle[*it].insert(horizon_angle); 
                            }
                        }
                    } else {
                        scan_horizon_angle[*it].insert(horizon_angle); 
                    }
                }
            }
            if (scan_horizon_angle.empty()) return false; 
            std::cout<<"scan_angle_set size:"<<scan_angle_set.size()<<", ";
            for (auto it = scan_angle_set.begin(); it != scan_angle_set.end(); it++) {
                std::cout<<*it<<", ";
            }
            std::cout<<std::endl;
            // step2: 从点最多的scan开始循环求解水平分辨率  
            /**
             *@todo 改成排序  while 循环从点最多的scan开始求解 一直到求解完成 
             */
            float max_num_scan = scan_horizon_angle.begin()->first;
            int max_num = scan_horizon_angle.begin()->second.size(); 
            for (auto it = ++scan_horizon_angle.begin(); it != scan_horizon_angle.end(); it++) {
                if (max_num < (int)it->second.size()) {
                    max_num = it->second.size(); 
                    max_num_scan = it->first;
                }
            }
            std::multiset<float> horizon_incre_angle;
            float last_angle = *scan_horizon_angle[max_num_scan].begin(); 
            for (auto it = ++scan_horizon_angle[max_num_scan].begin(); 
                it != scan_horizon_angle[max_num_scan].end(); it++) {
                    horizon_incre_angle.insert(*it - last_angle);
                    last_angle = *it; 
            }
            // 双指针找到占比超过阈值的数据    并求解平均值 
            uint16_t good_num_thresh = horizon_incre_angle.size() * 0.95;
            uint16_t min_num_thresh = horizon_incre_angle.size() * 0.8;
            auto ptr_front = horizon_incre_angle.begin();
            auto ptr_back = horizon_incre_angle.begin();
            uint16_t idx_front = 0, idx_back = 0; 
            uint16_t max_len = 0; 
            auto best_ptr_front = ptr_front, best_ptr_back = ptr_back;
            uint16_t best_idx_front = 0, best_idx_back = 0; 

            while (ptr_back != horizon_incre_angle.end()) {
                // 认为误差范围小于 0.02度 
                if (*ptr_back - *ptr_front < 0.02) {
                    if (idx_back - idx_front > max_len) {
                        max_len = idx_back - idx_front; 
                        best_idx_front = idx_front;
                        best_ptr_front = ptr_front; 
                        best_idx_back = idx_back;
                        best_ptr_back = ptr_back;
                        if (max_len > good_num_thresh) {
                            LOG(INFO) << "Find primary horizon resolution data! rate over 95% !"; 
                            break;
                        }
                    }
                    ptr_back++; 
                    idx_back++;
                } else {
                    ptr_front++;
                    idx_front++; 
                }
            }
            if (max_len < min_num_thresh) return false;
            LOG(INFO) << "Find primary horizon resolution data! "; 
            LOG(INFO) <<"index from : "<<best_idx_front <<" to " << best_idx_back; 
            // 遍历求解均值  
            double avg = 0;
            uint16_t N = 0; 
            auto ptr = best_ptr_front; 
            for (uint16_t idx = best_idx_front; idx <= best_idx_back; idx++) {
                // LOG(INFO)<<*ptr<<", ";
                avg += (*ptr - avg) / (N + 1);
                N++; 
                ptr++; 
            }
            option_.line_num_ = scan_angle_set.size(); 
            option_.horizon_angle_resolution_ = avg;  
            HORIZON_POINT_NUM_ = std::ceil(360 / option_.horizon_angle_resolution_) + 1;   
            LOG(INFO) << "LIDAR intrinsic calib done!";
            LOG(INFO) << "line_num_:"<< option_.line_num_;
            LOG(INFO) << "horizon_angle_resolution_:"<<option_.horizon_angle_resolution_; 
            LOG(INFO) << "horizon_point_num:"<<HORIZON_POINT_NUM_; 
            return true; 
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        inline uint16_t const& GetModelLineNum() const {
            return option_.line_num_; 
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        inline uint16_t const& GetModelHorizonPointNum() const {
            return HORIZON_POINT_NUM_; 
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        inline float const& GetModelHorizonPointResolution() const {
            return option_.horizon_angle_resolution_; 
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        int16_t GetRingID(_PointT const& point) {
            if (point.ring == -1) {
                return calcRingID(point);
            }
            return point.ring; 
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        std::pair<uint16_t, float> GetColumn(_PointT const& point) {
            float horizon_angle = atan2(point.x, point.y) * 180 / M_PI;    // atan2(y, x) 
            if (horizon_angle < 0) {
                horizon_angle += 360; 
            }
            //uint16_t column_idx = (uint16_t)(horizon_angle / option_.horizon_angle_resolution_ );
            uint16_t column_idx = round(horizon_angle / option_.horizon_angle_resolution_ );
            // float horizon_angle = atan2(point.x, point.y) * 180 / M_PI;
            // static float ang_res_x = 360.0/float(HORIZON_POINT_NUM_);
            // int columnIdn = -round((horizon_angle-90.0)/ang_res_x) + HORIZON_POINT_NUM_/2;
            // if (columnIdn >= HORIZON_POINT_NUM_)
            //     columnIdn -= HORIZON_POINT_NUM_;

            // if (columnIdn < 0 || columnIdn >= HORIZON_POINT_NUM_)
            //     return std::make_pair(0, -1); 

            return std::make_pair(column_idx, horizon_angle); 
        }

    protected:
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void init() {
            HORIZON_POINT_NUM_ = std::ceil(360 / option_.horizon_angle_resolution_) + 1;   
            std::cout<<"lidar model HORIZON_POINT_NUM_:"<<HORIZON_POINT_NUM_<<std::endl;
            if (option_.line_num_ == 16) {
                vertical_angle_resolution_ = 2.0;
                angle_bottom_ = 15.0 + 0.1;
            } else if (option_.line_num_ == 32) {
                vertical_angle_resolution_ = 41.33 / float(option_.line_num_ - 1);
                angle_bottom_ = 30.0 + 0.67;
            }
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        int16_t calcRingID(_PointT const& point) {
            int16_t row_id = 0;     
            double vertical_angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
            // 64线雷达  线束分布 不均匀 
            if (option_.line_num_ == 64) { 
                // VLP-64
                if (vertical_angle >= -8.83) {
                    row_id = static_cast<int>((2 - vertical_angle) * 3.0 + 0.5);
                } else {
                    row_id = static_cast<int>(option_.line_num_ / 2) 
                        + static_cast<int>((-8.83 - vertical_angle) * 2.0 + 0.5);
                }
                if (vertical_angle > 2 || vertical_angle < -24.33 || row_id > 50 || row_id < 0) {
                    return -1;
                }
            } else if (option_.line_num_ == 16 || option_.line_num_ == 32) {
                // 16、32线雷达 线束分布一般是均匀的
                row_id = static_cast<int>((vertical_angle + angle_bottom_) / vertical_angle_resolution_);
                if (row_id < 0 || row_id >= option_.line_num_) {
                    return -1;
                }
            } else {
                throw "wrong scan number";
            }
            return row_id; 
        }

    private:
        Option option_;  
        float angle_bottom_ = 0, vertical_angle_resolution_ = 0; 
        uint16_t HORIZON_POINT_NUM_ = 0;   // 每一个scan 包含的点数
}; // class 
} // namespace 
}