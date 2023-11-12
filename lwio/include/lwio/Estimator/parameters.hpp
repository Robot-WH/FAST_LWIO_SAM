#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

namespace lwio {
namespace estimator {

static const Eigen::Vector3d G{0.0, 0.0, 9.8};


enum SIZE_PARAMETERIZATION {
    SIZE_POSE = 7,
    SIZE_SPEEDBIAS = 9,
    SIZE_FEATURE = 1
};

enum BaseStateOrder {
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

// 扩展的状态  如外参、时间戳偏移...
enum ExpandStateOrder {
};

enum NoiseOrder {
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
}
}
