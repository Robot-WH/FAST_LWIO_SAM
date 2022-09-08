/*
 * @Copyright(C): Your Company
 * @FileName: 文件名
 * @Author: 作者
 * @Version: 版本
 * @Date: 2022-03-01 16:02:32
 * @Description: 
 * @Others: 
 */
#pragma once 
#include "Map/LocalMap.hpp"
#include "Map/PointcloudAreaLocalMap.hpp"
#include "Map/PointcloudSlidingLocalMap.hpp"

namespace Slam3D {

    template<typename _PointType, typename... _ParamType>
    static std::unique_ptr<PointCloudLocalMapBase<_PointType>> make_localMap(
            std::string const& type_name, std::string const& map_name, _ParamType... param) {
        std::unique_ptr<PointCloudLocalMapBase<_PointType>> local_map;  
        if (type_name == "time_sliding_window") {
            return std::unique_ptr<PointCloudLocalMapBase<_PointType>>(
                    new PointCloudSlidingLocalMap<_PointType>(map_name, param...));
        } else if (type_name == "space_sliding_window") {
        }
    }
} // namespace Factory 
