/*
 * @Copyright(C): 
 * @Author: lwh
 * @Date: 2022-04-09 17:00:25
 * @Description:  基于距离图像的点云分割 
 * @Others: 仅限于垂直&水平角度线性分布的雷达，貌似目前只有16线雷达满足，ref.legoloam 
 */

#pragma once
#include "Common/pcl_type.h"
#include "../LidarModel/RotatingLidarModel.hpp"
#include "../GroundDetect/AngleBasedGroundDetect.hpp"
#include <opencv/cv.hpp>
namespace Slam3D {
namespace Algorithm {

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: 基于距离图像的点云分割 
 * @details: 投影为距离图像进行地面分割以及聚类滤波等  用于旋转雷达 
 */    
template<typename _PointT>
class PointCloudRangeImageSegmentation {
    public:
        #define DEBUG 1
        // 配置参数 
        struct Option {
            uint16_t line_num_ = 16;    // 线数    默认16 
            uint16_t ground_line_num_ = 8;    // 地面线数
            float vertical_angle_resolution_ = 2.0; 
            float horizon_angle_resolution_ = 0.2;      // 0.2 10hz  0.386 19.3hz 

            float segment_angle_thresh_ = 0.17;   // 10度  10 * 3.14 / 180 = 0.17
            float clustered_num_thresh_ = 30;    // 聚类成立的数量阈数
            float min_valid_num_ = 10;    // 聚类成立的数量阈数
            float min_vertical_line_num_ = 5; 
        };
        // 聚类体信息 
        struct ClusterInfo {
            Eigen::Vector3d direction_max_value_{0, 0, 0};    // 3D聚类的主方向分布最大范围
            Eigen::Vector3d centre_{0, 0, 0}; // 聚类中心 
            Eigen::Vector3d direction_{0, 0, 0};  // 3D聚类的主方向
        };

        enum ClusterType{
            STABLE_POINT = 1,   // 若聚类后  聚类体的size足够大  那么认为是稳定的
            UNSTABLE_POINT,  // 聚类后，size 不够大，则认为是潜在的动态点 
            GROUND_POINT 
        };

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // 构造一：外部进行参数配置 然后传入参数进行初始化
        PointCloudRangeImageSegmentation(Option const& option) : option_(option){
            Init();
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // 构造二：传入参数文件地址  直接读参数文件初始化参数 
        PointCloudRangeImageSegmentation(std::string config_path) {}

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void Init() {
            std::pair<int8_t, int8_t> neighbor;
            // 上
            neighbor.first = 1;     
            neighbor.second = 0;
            neighbor_loc_.push_back(neighbor);
            // 左
            neighbor.first = 0;     
            neighbor.second = -1;  
            neighbor_loc_.push_back(neighbor);
            // 下
            neighbor.first = -1;
            neighbor.second = 0;
            neighbor_loc_.push_back(neighbor);
            // 右 
            neighbor.first = 0;
            neighbor.second = 1;
            neighbor_loc_.push_back(neighbor);

            SCAN_NUM_ = std::ceil(360 / option_.horizon_angle_resolution_);

            all_clustered_point_row = new uint16_t[option_.line_num_ * SCAN_NUM_];
            all_clustered_point_col = new uint16_t[option_.line_num_ * SCAN_NUM_];
            cluster_info_container_.reserve(2000); 
            // 旋转雷达模型 
            typename RotatingLidarModel<_PointT>::Option lidar_model_option;
            lidar_model_option.line_num_ = option_.line_num_;
            lidar_model_option.horizon_angle_resolution_ = option_.horizon_angle_resolution_;
            lidar_model_ptr_.reset(new RotatingLidarModel<_PointT>(lidar_model_option));
            // 地面提取
            typename AngleBasedGroundDetect<_PointT>::Option ground_detect_option;
            ground_detect_option.line_num_ = option_.line_num_;
            ground_detect_option.ground_line_num_ = option_.ground_line_num_;
            ground_detect_option.horizon_angle_resolution_ = option_.horizon_angle_resolution_;
            ground_detect_.reset(new AngleBasedGroundDetect<_PointT>(ground_detect_option));  

            resetParameters(); 
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void resetParameters() {
            range_image_ = cv::Mat(option_.line_num_, SCAN_NUM_, CV_32F, cv::Scalar::all(FLT_MAX));
            index_image_ = cv::Mat(option_.line_num_, SCAN_NUM_, CV_32S, cv::Scalar::all(-1));
            label_image_ = cv::Mat(option_.line_num_, SCAN_NUM_, CV_32S, cv::Scalar::all(0));
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        virtual void Process(pcl::PointCloud<_PointT> const& cloud_in) {
            projectPointcloud(cloud_in); 
            ground_detect_->GroundDetect(cloud_in, index_image_); // 地面检测 
            cloudSegmentation(cloud_in);  // 聚类分割 
            // 提取点云 
            ground_pointcloud_.clear(); 
            stable_pointcloud_.clear();
            unstable_pointcloud_.clear();
            outlier_pointcloud_.clear(); 
            for (uint16_t i = 0; i < option_.line_num_; ++i) {
                for (uint16_t j = 0; j < SCAN_NUM_; ++j) {
                    int index = index_image_.at<int32_t>(i, j);
                    if (index == -1) continue; 
                    if (label_image_.at<int32_t>(i, j) == ClusterType::GROUND_POINT) {
                        ground_pointcloud_.push_back(cloud_in.points[index]);
                    } else if (label_image_.at<int32_t>(i, j) == ClusterType::STABLE_POINT) {
                        stable_pointcloud_.push_back(cloud_in.points[index]);
                    } else if (label_image_.at<int32_t>(i, j) == ClusterType::UNSTABLE_POINT) {
                        unstable_pointcloud_.push_back(cloud_in.points[index]);
                    } else if (label_image_.at<int32_t>(i, j) == 999999) {
                        outlier_pointcloud_.push_back(cloud_in.points[index]);
                    }
                }
            }
            resetParameters(); 
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        pcl::PointCloud<_PointT> const& GetGroundPoints() {
            return ground_pointcloud_;
        }   

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        pcl::PointCloud<_PointT> const& GetStablePoints() {
            return stable_pointcloud_;
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        pcl::PointCloud<_PointT> const& GetUnStablePoints() {
            return unstable_pointcloud_;
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        pcl::PointCloud<_PointT> const& GetOutlierPoints() {
            return outlier_pointcloud_;
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        std::vector<ClusterInfo> const& GetClusterInfo() {
            return cluster_info_container_;
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void SetRingFlag(bool flag) {
           lidar_model_ptr_->SetCalcRing(flag);
        }

    protected:
        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief:  构造点云图像 
         */            
        void projectPointcloud(pcl::PointCloud<_PointT> const& laserCloudIn) {
            float range;
            int16_t rowIdn;
            int32_t columnIdn;
            _PointT thisPoint;
            int32_t cloudSize = laserCloudIn.points.size();
            // std::cout<<"laserCloudIn size: "<<cloudSize<<std::endl;
            int num = 0; 
            for (int32_t i = 0; i < cloudSize; ++i) {
                thisPoint.x = laserCloudIn.points[i].x;
                thisPoint.y = laserCloudIn.points[i].y;
                thisPoint.z = laserCloudIn.points[i].z;
                thisPoint.range = laserCloudIn.points[i].range;
                // 点云模型 求解激光点坐标 
                rowIdn = lidar_model_ptr_->GetRingID(thisPoint); 
                if (rowIdn < 0) continue; 
                columnIdn = lidar_model_ptr_->GetColumn(thisPoint); 
                if (columnIdn < 0) continue;  
                range_image_.at<float>(rowIdn, columnIdn) = thisPoint.range;
                index_image_.at<int32_t>(rowIdn, columnIdn) = i;  
                // num++; 
            }
            // std::cout<<"index  num: "<<num<<std::endl;
            # if(DEBUG == 1)
                showRangeImage();
            #endif 
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void showRangeImage() {
            // 构造opencv mat 
            cv::Mat img(option_.line_num_, SCAN_NUM_, CV_8UC1, cv::Scalar(255));
            // 转为灰度     0-黑   255-白 
            for (uint16_t row = 0; row < option_.line_num_; row++) {
                for (uint16_t col = 0; col < SCAN_NUM_; col++) {
                    // 越远越白
                   img.at<uint8_t>(row, col) = (uint8_t)(range_image_.at<float>(row, col) * 255 / 200);
                }
            }
            std::cout<<"showRangeImage"<<std::endl;
            cv::imshow("range image", img);
            cv::waitKey(1);  
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void showLabelImage() {
            // // 构造opencv mat 
            // cv::Mat img(option_.line_num_, SCAN_NUM_, CV_8UC1, cv::Scalar(255));
            // // 转为灰度     0-黑   255-白 
            // for (uint16_t row = 0; row < option_.line_num_; row++) {
            //     for (uint16_t col = 0; col < SCAN_NUM_; col++) {
            //         // 越远越白
            //        img.at<uint8_t>(row, col) = (uint8_t)(range_image_.at<float>(row, col) * 255 / 200);
            //     }
            // }
            // cv::imshow("label image", img);
            // cv::waitKey(1);  
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: 点云分割  
         */        
        void cloudSegmentation(pcl::PointCloud<_PointT> const& cloud_in) {
            // 先将地面分离出来 
            cv::Mat const& ground_image =  ground_detect_->GetGroundImage(); 
            // 地面 以及 无效点 标记 不参与聚类  
            for (uint16_t i = 0; i < option_.line_num_; ++i) {
                for (uint16_t j = 0; j < SCAN_NUM_; ++j) {
                    if (i < option_.ground_line_num_) {
                        if (ground_image.at<int8_t>(i, j) == 1) {
                            label_image_.at<int>(i, j) = ClusterType::GROUND_POINT;
                            continue;  
                        }
                    }
                    if (index_image_.at<int32_t>(i, j) == -1) {
                        label_image_.at<int>(i, j) = -1;
                    }
                }
            }
            cluster_info_container_.clear(); 
            //std::cout<<"label_image_: "<<label_image_<<std::endl;
            for (uint16_t i = 0; i < option_.line_num_; ++i) {
                for (uint16_t j = 0; j < SCAN_NUM_; ++j) {
                    rangeImageBfsCluster(i, j, cloud_in);
                }
            }
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /**
         * @brief: BFS 对距离图像进行聚类
         * @param row
         * @param col
         */        
        void rangeImageBfsCluster(uint16_t const& row, uint16_t const& col, 
                pcl::PointCloud<_PointT> const& cloud_in) {
            if (label_image_.at<int>(row, col) != 0) {
                return;  
            }
            float d1, d2, alpha, angle;
            uint16_t centre_row, centre_col, curr_row, curr_col;    // 图像坐标 
            bool line_occupy_flag[option_.line_num_] = {false};
            std::deque<uint16_t> bfs_queue_row;
            std::deque<uint16_t> bfs_queue_col;
            bfs_queue_row.push_front(row);
            bfs_queue_col.push_front(col);
            std::vector<uint32_t> cluster_index;
            cluster_index.reserve(option_.line_num_ * SCAN_NUM_); 
            cluster_index.push_back(index_image_.at<uint32_t>(row, col)); 

            all_clustered_point_row[0] = row;
            all_clustered_point_col[0] = col;
            uint32_t all_clustered_point_num = 1;
            // 保存聚类边界的图像坐标 上下左右  
            std::vector<uint16_t> cluster_area_row = std::vector<uint16_t>(4, row);
            std::vector<uint16_t> cluster_area_col = std::vector<uint16_t>(4, col);

            while (!bfs_queue_row.empty()) {
                centre_row = bfs_queue_row.front();
                centre_col = bfs_queue_col.front();
                bfs_queue_row.pop_front();
                bfs_queue_col.pop_front();
                label_image_.at<int>(centre_row, centre_col) = ClusterType::STABLE_POINT;   // 先默认赋值为稳定点 
                // 周围的邻居   上左下右  
                for (auto iter = neighbor_loc_.begin(); iter != neighbor_loc_.end();
                    ++iter) {
                    curr_row = centre_row + (*iter).first;
                    curr_col = centre_col + (*iter).second;

                    if (curr_row < 0 || curr_row >= option_.line_num_) continue;    
                     // Y坐标超过图像边界  那么应该进入图像的另一边
                    if (curr_col < 0) curr_col = SCAN_NUM_ - 1;
                    if (curr_col >= SCAN_NUM_) curr_col = 0;
                    if (label_image_.at<int>(curr_row, curr_col) != 0) continue;
                    // // 下面是更新聚类边界
                    // // 右移
                    // if ((*iter).second == 1) {
                    //     //std::cout<<"curr_col: "<<curr_col<<", cluster_area_col[3]: "<<cluster_area_col[3]<<std::endl;
                    //     if (curr_col == 0 || curr_col == cluster_area_col[3] + 1) {
                    //         cluster_area_row[3] = curr_row;
                    //         cluster_area_col[3] = curr_col;
                    //     } 
                    // } else if ((*iter).second == -1) { // 左移
                    //     //std::cout<<"curr_col: "<<curr_col<<", cluster_area_col[2]: "<<cluster_area_col[2]<<std::endl;
                    //     if (curr_col == SCAN_NUM_ - 1 || curr_col == cluster_area_col[2] - 1) {
                    //         cluster_area_row[2] = curr_row;
                    //         cluster_area_col[2] = curr_col;
                    //     }
                    // } else if ((*iter).first == 1)  { // 上移
                    //     //std::cout<<"curr_row: "<<curr_row<<", cluster_area_col[0]: "<<cluster_area_col[0]<<std::endl;
                    //     if (curr_row == cluster_area_row[0] + 1) {
                    //         cluster_area_row[0] = curr_row;
                    //         cluster_area_col[0] = curr_col;
                    //     }
                    // } else if ((*iter).first == -1)  { // 下移
                    //     //std::cout<<"curr_row: "<<curr_col<<", cluster_area_col[0]: "<<cluster_area_col[0]<<std::endl;
                    //     if (curr_row == cluster_area_row[1] - 1) {
                    //         cluster_area_row[1] = curr_row;
                    //         cluster_area_col[1] = curr_col;
                    //     }
                    // }
                    // 通过距离图像判断是否是同类   
                    d1 = std::max(range_image_.at<float>(centre_row, centre_col),
                                range_image_.at<float>(curr_row, curr_col));
                    d2 = std::min(range_image_.at<float>(centre_row, centre_col),
                                range_image_.at<float>(curr_row, curr_col));
                    // 角度的变化量 
                    if ((*iter).first == 0) {
                        alpha = option_.vertical_angle_resolution_; 
                    } else {
                        alpha = option_.horizon_angle_resolution_;
                    }
                    // angle 越小 说明距离越远  越不可能是一类  
                    angle = atan2(d2 * sin(alpha), (d1 - d2 * cos(alpha)));
                    // 说明是一类  
                    if (angle > option_.segment_angle_thresh_) {
                        // 保存该点  等待bfs聚类 
                        bfs_queue_row.push_back(curr_row);
                        bfs_queue_col.push_back(curr_col);
                        cluster_index.push_back(index_image_.at<uint32_t>(curr_row, curr_col)); 
                        label_image_.at<int>(curr_row, curr_col) = ClusterType::STABLE_POINT;
                        line_occupy_flag[curr_row] = true;
                        // 临时存放本次聚类的全部点的坐标 
                        all_clustered_point_row[all_clustered_point_num] = curr_row;
                        all_clustered_point_col[all_clustered_point_num] = curr_col;
                        ++all_clustered_point_num;
                    }
                }
            }
            // 如果聚类后 点数太小 < 30   认为是小物体   或者很稀疏   
            bool feasibleSegment = false;
            if (all_clustered_point_num >= option_.clustered_num_thresh_) {
                feasibleSegment = true;
            } // 对于竖直的线 需要另外考虑
            else if (all_clustered_point_num >= option_.min_valid_num_) {
                // 检查垂直的线束 
                int lineCount = 0;
                for (size_t i = 0; i < option_.line_num_; ++i) {
                    if (line_occupy_flag[i] == true) {
                        ++lineCount;
                    }
                }
                if (lineCount >= option_.min_vertical_line_num_) {
                    feasibleSegment = true;
                }
            }

            if (feasibleSegment == true) {
                // // std::cout<<"all_clustered_point_num: "<<all_clustered_point_num<<std::endl;
                // // std::cout<<"left row: "<<cluster_area_row[2]<<", col: "<<cluster_area_col[2]<<std::endl;
                // // std::cout<<"right row: "<<cluster_area_row[3]<<", col: "<<cluster_area_col[3]<<std::endl;
                // // std::cout<<"up row: "<<cluster_area_row[0]<<", col: "<<cluster_area_col[0]<<std::endl;
                // // std::cout<<"down row: "<<cluster_area_row[1]<<", col: "<<cluster_area_col[1]<<std::endl;
                // // 粗略计算聚类的size，根据size 将聚类分类为稳定和可能动态两类
                // _PointT area_left =cloud_in.points[index_image_.at<int32_t>(cluster_area_row[2], cluster_area_col[2])]; 
                // _PointT area_right =cloud_in.points[index_image_.at<int32_t>(cluster_area_row[3], cluster_area_col[3])]; 
                // _PointT area_up =cloud_in.points[index_image_.at<int32_t>(cluster_area_row[0], cluster_area_col[0])]; 
                // _PointT area_down =cloud_in.points[index_image_.at<int32_t>(cluster_area_row[1], cluster_area_col[1])]; 
                // Eigen::Vector3d lenght_vec = {area_left.x - area_right.x, area_left.y - area_right.y, area_left.z - area_right.z};
                // Eigen::Vector3d height_vec = {area_up.x - area_down.x, area_up.y - area_down.y, area_up.z - area_down.z};
                // //std::cout<<"lenght: "<<lenght_vec.norm()<<", height: "<<height_vec.norm()<<std::endl;
                // // if (lenght_vec.norm() < 8 && height_vec.norm() < 3) {
                ClusterInfo info;
                pca(cloud_in, cluster_index, info); 
                if (info.direction_max_value_[2] < 10) {
                    for (uint32_t i = 0; i < all_clustered_point_num; ++i) {
                        label_image_.at<int>(all_clustered_point_row[i], all_clustered_point_col[i]) = ClusterType::UNSTABLE_POINT;
                    }
                    cluster_info_container_.push_back(std::move(info));
                }
            } else {
                // 不好的聚类  进行标记   
                for (uint32_t i = 0; i < all_clustered_point_num; ++i) {
                    label_image_.at<int>(all_clustered_point_row[i], all_clustered_point_col[i]) = 999999;
                }
            }
        }

        //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        void pca(pcl::PointCloud<_PointT> const& cloud_in, 
                std::vector<uint32_t> const& index,  ClusterInfo &info) {
            std::vector<Eigen::Vector3d> points;
            std::vector<Eigen::Vector3d> decentration_points;   // 去中心化
            info.centre_ = {0, 0, 0};
            uint16_t size = index.size();  
            points.reserve(size);
            decentration_points.reserve(size);

            for (int32_t const& i : index) {
                Eigen::Vector3d point(cloud_in.points[i].x, cloud_in.points[i].y, cloud_in.points[i].z);
                info.centre_ += point;
                points.push_back(std::move(point));
            }
            info.centre_ /= size;
            // 计算协方差矩阵
            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();
            for (int32_t i = 0; i < size; i++) {
                Eigen::Vector3d tmpZeroMean = points[i] - info.centre_;
                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                decentration_points.push_back(std::move(tmpZeroMean));
            }
            covMat /= size;  
            // 特征分解 
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);
            info.direction_max_value_ = {0, 0, 0};
            auto eigen_vector = saes.eigenvectors(); 
            // 将各个点向主方向投影
            for (uint8_t i = 2; i < 3; i++) {
                Eigen::Vector3d direction = eigen_vector.col(i);
                for (int32_t j = 0; j < size; j++) {
                    double value = decentration_points[j].transpose() * direction; 
                    if (value > info.direction_max_value_[i]) {
                        info.direction_max_value_[i] = value;  
                    }   
                }
            }
        }

    private:
        Option option_;  
        std::unique_ptr<RotatingLidarModel<_PointT>> lidar_model_ptr_;  
        std::unique_ptr<AngleBasedGroundDetect<_PointT>> ground_detect_;    // 地面检测
        cv::Mat range_image_;
        cv::Mat index_image_;  
        cv::Mat label_image_;
        pcl::PointCloud<_PointT> ground_pointcloud_;
        pcl::PointCloud<_PointT> stable_pointcloud_;
        pcl::PointCloud<_PointT> unstable_pointcloud_;
        pcl::PointCloud<_PointT> outlier_pointcloud_; 
        std::vector<std::pair<uint8_t, uint8_t> > neighbor_loc_;
        std::vector<ClusterInfo> cluster_info_container_;
        uint16_t* all_clustered_point_row;
        uint16_t* all_clustered_point_col;
        uint16_t SCAN_NUM_;   
}; // class 

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief: 工厂函数
 * @param file_path 参数文件地址  
 */    
template<typename _T>
std::unique_ptr<PointCloudRangeImageSegmentation<_T>> Create(std::string file_path) {

}
}
} // namespace 


