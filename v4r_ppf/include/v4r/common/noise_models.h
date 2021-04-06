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
 * @file model.h
 * @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at), Aitor Aldoma (aldoma@acin.tuwien.ac.at)
 * @date 2015
 * @brief
 *
 */
#pragma once

#include <pcl/common/angles.h>
#include <pcl/common/common.h>
#include <pcl/common/io.h>

namespace v4r {

/**
 * @brief computes Kinect axial and lateral noise parameters for an organized point cloud
 * according to Nguyen et al., 3DIMPVT 2012.
 * It also computed depth discontinuites using PCL's organized edge detection algorithm and the distance of
 * each pixel to these discontinuites.
 * @author Thomas Faeulhammer, Aitor Aldoma
 * @date December 2015
 */
template <class PointT>
class NguyenNoiseModel {
 public:
  /**
   * @brief computes the point properties for each point (axial, lateral noise as well as distance to depth
   * discontinuities)
   * @param input
   * @param normals
   * @param focal_length
   * @return returns for each pixel lateral [idx=0] and axial [idx=1] as well as distance in pixel to closest depth
   * discontinuity [idx=2]
   */
  static std::vector<std::vector<float>> compute(const typename pcl::PointCloud<PointT>::ConstPtr &input,
                                                 const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
                                                 float focal_length = 525.f);

  /**
   * @brief compute the expected noise level for one pixel only
   * @param point for which to compute noise level
   * @param surface normal at this point
   * @param sigma_lateral in meters
   * @param sigma_axial in meters
   */
  static void computeNoiseLevel(const PointT &pt, const pcl::Normal &n, float &sigma_lateral, float &sigma_axial,
                                float focal_length = 525.f);
};
}  // namespace v4r
