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

#include <ppf/local_coordinate.h>

namespace ppf {

void LocalCoordinate::computeTransform(const Eigen::Vector3f& p, const Eigen::Vector3f& n,
                                       Eigen::Translation3f& translation, Eigen::Quaternionf& rotation) {
  float rotation_angle = std::acos(n.dot(Eigen::Vector3f::UnitX()));
  bool parallel_to_x = (n.y() == 0.0f && n.z() == 0.0f);
  Eigen::Vector3f rotation_axis =
      (parallel_to_x) ? (Eigen::Vector3f::UnitY()) : (n.cross(Eigen::Vector3f::UnitX()).normalized());
  rotation = Eigen::AngleAxisf(rotation_angle, rotation_axis);
  translation = Eigen::Translation3f(rotation * ((-1) * p));
}

void LocalCoordinate::computeTransform(const Eigen::Vector3f& p, const Eigen::Vector3f& n, Eigen::Affine3f& transform) {
  Eigen::Translation3f translation;
  Eigen::Quaternionf rotation;
  computeTransform(p, n, translation, rotation);
  transform = translation * rotation;
}

float LocalCoordinate::computeAngle(const Eigen::Vector3f& p) {
  float angle = std::atan2(-p(2), p(1));
  if (std::sin(angle) * p(2) < 0.0f)
    return angle;
  return -angle;
}

}  // namespace ppf
