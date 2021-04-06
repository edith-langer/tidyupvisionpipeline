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

#pragma once

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace ppf {

/// A structure representing local coordinate (LC) on a model.
///
/// For a given scene point that belong to an object instance, 6D pose of this instance can be unambiguously defined by
/// a model point (to which this scene point corresponds) and a rotation angle around their aligned normals. These two
/// values together are denoted as LC. See [VLLM18] page 5 for more details.
///
/// The structure provides several helper functions to compute transforms and angles related with local coordinates.
struct LocalCoordinate {
  uint32_t model_point_index1;  ///< Index of a point in model point cloud.
  uint32_t model_point_index2;
  float rotation_angle;        ///< Rotation angle around model point normal, in radians.

  using Vector = std::vector<LocalCoordinate>;

  /// Compute transform that puts point at origin and aligns normal with x-axis.
  /// This is a helper function to compute transforms T_s and T_m in [VLLM18] equation 3.
  static void computeTransform(const Eigen::Vector3f& point, const Eigen::Vector3f& normal, Eigen::Affine3f& transform);

  /// Compute transform that puts point at origin and aligns normal with x-axis.
  /// This is a helper function to compute transforms T_s and T_m in [VLLM18] equation 3.
  static void computeTransform(const Eigen::Vector3f& point, const Eigen::Vector3f& normal,
                               Eigen::Translation3f& translation, Eigen::Quaternionf& rotation);

  /// Compute angle for rotation around x-axis that puts point on xy-plane.
  /// This is a helper function to compute angles alpha_s and alpha_m in [VLLM18] figure 4.
  static float computeAngle(const Eigen::Vector3f& point);
};

}  // namespace ppf
