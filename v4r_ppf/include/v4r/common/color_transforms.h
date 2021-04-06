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
 * @file color_transforms.h
 * @date 2019
 * @brief
 *
 */

#pragma once
#include <pcl/point_cloud.h>
#include <v4r/common/time.h>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace v4r {

/**
 * @tparam PointT point type
 * @param src input cloud with color information
 * @param dst output image of the same size and depth as src.
 * @param code color space conversion code (see OpenCV's #ColorConversionCodes).
 */
template <typename PointT>
void convertColor(const pcl::PointCloud<PointT> &cloud, cv::OutputArray color_mat, int code) {
  std::vector<cv::Vec3b> rgb;
  rgb.reserve(cloud.size());
  for (const auto &p : cloud.points)
    rgb.emplace_back(p.r, p.g, p.b);

  cv::cvtColor(rgb, color_mat, code);
}
}  // namespace v4r