#include "lwio/system.h"
#include "SlamLib/PointCloud/Filter/voxel_grid.h"
#include "SlamLib/PointCloud/Filter/outlier_removal.h"
// #include "SlamLib/Common/point_type.h"

int main() {
    SlamLib::pointcloud::VoxelGridFilter<pcl::PointXYZI> vg; 
    SlamLib::pointcloud::OutlierRemovalFilter<pcl::PointXYZI> rm;
    // std::unique_ptr<lwio::System<pcl::PointXYZI>> system_ptr_; 
    // system_ptr_.reset(new lwio::System<pcl::PointXYZI>("dasasdsads"));
    std::unique_ptr<lwio::System<PointXYZIRDT>> system_ptr_; 
    system_ptr_.reset(new lwio::System<PointXYZIRDT>("dasasdsads"));

    return 0; 
}