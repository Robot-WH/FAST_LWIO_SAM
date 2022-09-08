
#pragma once

#include <cmath>
#include <eigen3/Eigen/Dense>

namespace Slam3D {

constexpr double CoefDegreeToRadian = M_PI / 180.;
constexpr double CoefRadianToDegree = 180. / M_PI;
constexpr double Gravity = 9.81;  

/**
 * @brief: 常用功能类 
 */
class Utility {
    public:
        // 对角度进行标准化     [-M_PI, M_PI]
        static void NormalizeAngle(double& angle) {        
            if(angle >= M_PI)
                angle -= 2.0*M_PI;
            if(angle < -M_PI)
                angle += 2.0*M_PI;
        }

        /**
         * @brief: 插值的模板函数
         * @details: 部分类型的插值需要进行特化  
         */    
        template<typename _T>
        static _T Interpolate(_T front_value, _T back_value, double const& front_time, 
                double const& back_time, double const& time) {
            _T interpolation_v;
            float front_coeff = (back_time - time) / (back_time - front_time);
            float back_coeff = (time - front_time) / (back_time - front_time);
            interpolation_v = front_coeff * front_value + back_coeff * back_value;  
            return interpolation_v;  
        } 

        /**
         * squared distance
         * @param p1
         * @param p2
         * @return
         */
        template<class PointType>
        static inline float CalcDist(const PointType &p1, const PointType &p2) {
            return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
        }

        static inline float CalcDist(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2) { 
            return (p1 - p2).squaredNorm(); 
        }
}; 

/**
 * @brief 保存轨迹数据  
 * 
 */
// static bool SaveTrajectory(Eigen::Matrix4d const& gt_odom, Eigen::Matrix4d const& est_odom, string const& directory_path,
//                            string const& file_1, string const& file_2) {
//     static std::ofstream ground_truth, est_path;
//     static bool is_file_created = false;
//     if (!is_file_created) {
//         if (!FileManager::CreateDirectory(WORK_SPACE_PATH + directory_path))
//             return false;
//         if (!FileManager::CreateFile(ground_truth, WORK_SPACE_PATH + file_1))
//             return false;
//         if (!FileManager::CreateFile(est_path, WORK_SPACE_PATH + file_2))
//             return false;
//         is_file_created = true;
//     }

//     for (int i = 0; i < 3; ++i) {
//         for (int j = 0; j < 4; ++j) {
//             ground_truth << gt_odom(i, j);
//             est_path << est_odom(i, j);
//             if (i == 2 && j == 3) {
//                 ground_truth << std::endl;
//                 est_path << std::endl;
//             } else {
//                 ground_truth << " ";
//                 est_path << " ";
//             }
//         }
//     }

//     return true;
// }

// /**
//  * @brief 保存数据
//  * 
//  */
// static bool SaveDataCsv(string const& Directory_path, string const& file_path, 
//                         vector<double> const& datas, vector<string> const& labels) {
//     static std::ofstream out;
//     static bool is_file_created = false;

//     if (!is_file_created) {
//         if (!FileManager::CreateDirectory(WORK_SPACE_PATH + Directory_path))
//             return false;
//         if (!FileManager::CreateFile(out, WORK_SPACE_PATH + file_path))
//             return false;
//         is_file_created = true;
//         // 写标签
//         for(auto const& label:labels) {
//            out << label << ',';  
//         }
//         out << endl;
//     }

//     for(auto const& data:datas) {
//        out << data << ',';
//     }

//     out << endl;
//     return true;
// }
}


