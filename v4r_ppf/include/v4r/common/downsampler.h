/****************************************************************************
**
** Copyright (C) 2019 TU Wien, ACIN, Vision 4 Robotics (V4R) group
** Contact: v4r.acin.tuwien.ac.at
**
** This file is part of V4R
**
** V4R is distributed under dual licenses - GPLv3 or closed source.
**
** GNU General Public License Usage
** V4R is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published
** by the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
**
** V4R is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
** GNU General Public License for more details.
**
** Please review the following information to ensure the GNU General Public
** License requirements will be met: https://www.gnu.org/licenses/gpl-3.0.html.
**
**
** Commercial License Usage
** If GPL is not suitable for your project, you must purchase a commercial
** license to use V4R. Licensees holding valid commercial V4R licenses may
** use this file in accordance with the commercial license agreement
** provided with the Software or, alternatively, in accordance with the
** terms contained in a written agreement between you and TU Wien, ACIN, V4R.
** For licensing terms and conditions please contact office<at>acin.tuwien.ac.at.
**
**
** The copyright holder additionally grants the author(s) of the file the right
** to use, copy, modify, merge, publish, distribute, sublicense, and/or
** sell copies of their contributions without any restrictions.
**
****************************************************************************/

/**
 * @file downsampler.h
 * @author Thomas Faeulhammer, Sergey Alexandrov
 * @date 2019
 * @brief Downsampling methods
 *
 */

#pragma once
#include <pcl/point_cloud.h>
#include <boost/program_options.hpp>

namespace v4r {
struct DownsamplerParameter {
  float resolution_ = 0.02f;  //< resolution of the downsampled point cloud in meter
  enum class Method {
    VOXEL,     ///< uses a voxel grid for downsampling
    ADVANCED,  ///<< clusters points inside an octree based on surface normals. Each octree node can have multiple
               ///< output
               ///< points
    TWO_LEVEL_ADVANCED,       ///< as above, but downsampling is applied once again with bigger resolution
    UNIFORM                   ///< use uniform sampling
  } method_ = Method::VOXEL;  //< downsampling method
  /// Parameter of advanced downsampling method.
  /// If angular distance between point normal and cluster normal is below this threshold (degrees), the point is merged
  /// into the cluster.
  float advanced_angular_distance_thershold_ = 25.0f;

  DownsamplerParameter() = default;
  DownsamplerParameter(float resolution, Method method = Method::VOXEL) : resolution_(resolution), method_(method) {}

  /**
   * @brief init parameters
   * @param command_line_arguments (according to Boost program options library)
   * @param section_name section name of program options
   */
  void init(boost::program_options::options_description &desc, const std::string &section_name = "downsampling");
};

std::istream &operator>>(std::istream &, DownsamplerParameter::Method &);
std::ostream &operator<<(std::ostream &, const DownsamplerParameter::Method &);

class Downsampler {
 private:
  DownsamplerParameter param_;
  std::shared_ptr<std::vector<int>> indices_;  ///< extracted indices of the original input cloud

 public:
  explicit Downsampler(const DownsamplerParameter &param) : param_(param) {}

  /**
   * @brief method to downsample a point cloud
   * @tparam PointT point type
   * @param input input point cloud
   * @param param Parameter object for downsampling
   * @return downsampled point cloud
   */
  template <typename PointT>
  typename pcl::PointCloud<PointT>::Ptr downsample(const typename pcl::PointCloud<PointT>::ConstPtr input);

  /**
   * @brief get extracted indices of the original cloud
   * @return extracted indices of the original cloud
   */
  auto getExtractedIndices() const {
    return indices_;
  }

  /**
   * @brief set parameter for downsampling
   * @param param Parameter object
   */
  void setParameter(const DownsamplerParameter &param) {
    param_ = param;
  }
};

}  // namespace v4r
