/****************************************************************************
**
** Copyright (C) 2017 TU Wien, ACIN, Vision 4 Robotics (V4R) group
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
 * @file normals.h
 * @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
 * @date 2015
 * @brief
 *
 */
#pragma once

#include <pcl/pcl_base.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/program_options.hpp>

namespace v4r {

void transformNormals(const pcl::PointCloud<pcl::Normal>& normals_cloud,
                                  pcl::PointCloud<pcl::Normal>& normals_aligned, const Eigen::Matrix4f& transform);

void transformNormals(const pcl::PointCloud<pcl::Normal>& normals_cloud,
                                  pcl::PointCloud<pcl::Normal>& normals_aligned, const std::vector<int>& indices,
                                  const Eigen::Matrix4f& transform);

struct NormalEstimatorParameter {
  enum class Method { PCL_DEFAULT, PCL_INTEGRAL_NORMAL, Z_ADAPTIVE };

  float radius_ = 0.02f;                           ///< smoothings size.
  bool use_omp_ = true;                            ///< if true, uses openmp for surface normal estimation
  float smoothing_size_ = 10.f;                    ///< smoothings size.
  float max_depth_change_factor_ = 20.f * 0.001f;  ///<  depth change threshold for computing object borders
  bool use_depth_depended_smoothing_ = true;       ///< use depth depended smoothing
  int kernel_ = 5;                                 ///< kernel radius [px]
  bool adaptive_ = false;                          ///< Activate z-adaptive normals calcualation
  float kappa_ = 0.005125f;                        ///< gradient
  float d_ = 0.f;                                  ///< constant
  std::vector<int> kernel_radius_ = {
      3, 3, 3, 3, 4, 5, 6, 7};  ///< Kernel radius for each 0.5 meter interval (e.g. if 8 elements, then 0-4m)
  Method method_ = Method::PCL_DEFAULT;

  void init(boost::program_options::options_description& desc, const std::string& section_name = "normal_estimator");
};
std::istream& operator>>(std::istream& in, NormalEstimatorParameter::Method& style);
std::ostream& operator<<(std::ostream& out, const NormalEstimatorParameter::Method& style);

/**
 * @brief Normal computation method for a point cloud
 * @tparam PointT point type
 * @param param Parameter object for normal computation
 * @param cloud input point cloud
 * @param indices point indices for which to compute surface normals.
 * @return computed surface normals
 */
template <typename PointT>
pcl::PointCloud<pcl::Normal>::Ptr computeNormals(const typename pcl::PointCloud<PointT>::ConstPtr cloud,
                                                 const NormalEstimatorParameter& param,
                                                 const pcl::IndicesConstPtr indices = nullptr);
}  // namespace v4r
