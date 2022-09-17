FAST_LWIO_SAM

简介 本项目实现一个快速的融合odom、IMU和激光雷达的里程计，并结合GTSAM进行优化，本项目由多个子项目构成：

分支项目1 <faster_loam>

本工程仅用一个激光雷达，目标是可适配多种激光雷达以及更快的运行速度。

功能特点：

1、基于迭代体素ivox以及C++17多线程加速实现快速激光里程计。

2、在开源版ivox的基础上增加了主动点删除操作以及动态降采样。

3、代码遵循可复用、低耦合、易扩展等特性，同一个代码工程无需修改即可直接适配多种雷达，如旋转雷达velodyne或半固态雷达如livox。

4、实现了一种基于优化的点云匹配框架，优化算法可选用高斯牛顿优化和LM优化，经过测试，在特征数量约为2000个的情况下，平均每一帧匹配时间约为10ms，仅为使用ceres自带的优化算法时耗时的1/3，而效果几乎不变。使用高斯牛顿优化和LM优化在ICP上的效果基本相同，高斯牛顿优化更为简单直接，故优先选用。