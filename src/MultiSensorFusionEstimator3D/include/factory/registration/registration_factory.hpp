/*
 * @Copyright(C): Your Company
 * @FileName: 文件名
 * @Author: 作者
 * @Version: 版本
 * @Date: 2022-01-19 11:28:57
 * @Description:  匹配算法的工厂类  
 * @Others: 
 */
#ifndef _REGISTRATION_FACTORY_HPP_
#define _REGISTRATION_FACTORY_HPP_

#include "utility.hpp"

#include <pcl/registration/ndt.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>

#include <pclomp/ndt_omp.h>
#include <pclomp/gicp_omp.h>

namespace Factory {

using NDTOMP = pclomp::NormalDistributionsTransform<PointType, PointType>; 
using ndtOmpPtr = std::unique_ptr<NDTOMP>;  

/**
 * @brief: 
 * @details: 
 * @param {*}
 * @return {*}
 */
ndtOmpPtr make_ndtomp(double const& ndt_resolution,
                                                        double const& transformation_epsilon, 
                                                        int const& maximum_iterations, 
                                                        int const& num_threads,
                                                        std::string const& nn_search_method) {
    // TODO 检查 maximum_iterations， num_threads 是否大于 0  
    ndtOmpPtr ndt_omp(new NDTOMP());
    if(num_threads > 0) {
      ndt_omp->setNumThreads(num_threads);      // 3
    }
    ndt_omp->setTransformationEpsilon(transformation_epsilon);    // 步长
    ndt_omp->setMaximumIterations(maximum_iterations);
    ndt_omp->setResolution(ndt_resolution);
    // 匹配方法
    if(nn_search_method == "KDTREE") {
      ndt_omp->setNeighborhoodSearchMethod(pclomp::KDTREE);
    } else if (nn_search_method == "DIRECT1") {
      ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT1);
    } else {
      ndt_omp->setNeighborhoodSearchMethod(pclomp::DIRECT7);
    } 
    return std::move(ndt_omp); 
}

// boost::shared_ptr<pcl::Registration<pcl::PointXYZI, pcl::PointXYZI>> select_registration_method(ros::NodeHandle& pnh) {
//   using PointT = pcl::PointXYZI;
 
//   // select a registration method (ICP, GICP, NDT)
//   std::string registration_method = pnh.param<std::string>("registration_method", "NDT_OMP");
//   if(registration_method == "ICP") {
//     std::cout << "registration: ICP" << std::endl;
//     boost::shared_ptr<pcl::IterativeClosestPoint<PointT, PointT>> icp(new pcl::IterativeClosestPoint<PointT, PointT>());
//     icp->setTransformationEpsilon(pnh.param<double>("transformation_epsilon", 0.01));
//     icp->setMaximumIterations(pnh.param<int>("maximum_iterations", 64));
//     icp->setUseReciprocalCorrespondences(pnh.param<bool>("use_reciprocal_correspondences", false));
//     return icp;
//   } else if(registration_method.find("GICP") != std::string::npos) {
//     if(registration_method.find("OMP") == std::string::npos) {
//       std::cout << "registration: GICP" << std::endl;
//       boost::shared_ptr<pcl::GeneralizedIterativeClosestPoint<PointT, PointT>> gicp(new pcl::GeneralizedIterativeClosestPoint<PointT, PointT>());
//       gicp->setTransformationEpsilon(pnh.param<double>("transformation_epsilon", 0.01));
//       gicp->setMaximumIterations(pnh.param<int>("maximum_iterations", 64));
//       gicp->setUseReciprocalCorrespondences(pnh.param<bool>("use_reciprocal_correspondences", false));
//       gicp->setCorrespondenceRandomness(pnh.param<int>("gicp_correspondence_randomness", 20));
//       gicp->setMaximumOptimizerIterations(pnh.param<int>("gicp_max_optimizer_iterations", 20));
//       return gicp;
//     } else {
//       std::cout << "registration: GICP_OMP" << std::endl;
//       boost::shared_ptr<pclomp::GeneralizedIterativeClosestPoint<PointT, PointT>> gicp(new pclomp::GeneralizedIterativeClosestPoint<PointT, PointT>());
//       gicp->setTransformationEpsilon(pnh.param<double>("transformation_epsilon", 0.01));
//       gicp->setMaximumIterations(pnh.param<int>("maximum_iterations", 64));
//       gicp->setUseReciprocalCorrespondences(pnh.param<bool>("use_reciprocal_correspondences", false));
//       gicp->setCorrespondenceRandomness(pnh.param<int>("gicp_correspondence_randomness", 20));
//       gicp->setMaximumOptimizerIterations(pnh.param<int>("gicp_max_optimizer_iterations", 20));
//       return gicp;
//     }
//   } else {
//     // 如果find查找不到  则会返回string::npos   
//     if(registration_method.find("NDT") == std::string::npos ) {
//       std::cerr << "warning: unknown registration type(" << registration_method << ")" << std::endl;
//       std::cerr << "       : use NDT" << std::endl;
//     }
    
//     double ndt_resolution = pnh.param<double>("ndt_resolution", 0.5);
//     // 如果没有使用  OMP   
//     if(registration_method.find("OMP") == std::string::npos) {
//       std::cout << "registration: NDT " << ndt_resolution << std::endl;
//       boost::shared_ptr<pcl::NormalDistributionsTransform<PointT, PointT>> ndt(new pcl::NormalDistributionsTransform<PointT, PointT>());
//       ndt->setTransformationEpsilon(pnh.param<double>("transformation_epsilon", 0.01));      // 迭代布长
//       ndt->setMaximumIterations(pnh.param<int>("maximum_iterations", 64));              
//       ndt->setResolution(ndt_resolution);                                                    // 设置网格化立体的边长 
//       return ndt;
//     } else { 
//       int num_threads = pnh.param<int>("ndt_num_threads", 0);
//       // 对于NDT_OMP来说还有个 nn_search_method的参数      默认  DIRECT7
//       std::string nn_search_method = pnh.param<std::string>("ndt_nn_search_method", "DIRECT7");
//       std::cout << "registration: NDT_OMP " << nn_search_method << " " << ndt_resolution << " (" << num_threads << " threads)" << std::endl;
//       boost::shared_ptr<pclomp::NormalDistributionsTransform<PointT, PointT>> ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
//       if(num_threads > 0) {
//         ndt->setNumThreads(num_threads);      // 3
//       }
//       ndt->setTransformationEpsilon(pnh.param<double>("transformation_epsilon", 0.01));    // 步长
//       ndt->setMaximumIterations(pnh.param<int>("maximum_iterations", 64));
//       ndt->setResolution(ndt_resolution);
//       // 匹配方法
//       if(nn_search_method == "KDTREE") {
//         ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
//       } else if (nn_search_method == "DIRECT1") {
//         ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
//       } else {
//         ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
//       } 
//       return ndt;
//     }
//   }  

//   return nullptr;
// }
}
#endif