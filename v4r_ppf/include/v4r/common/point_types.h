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
 * @file main.cpp
 * @author Johann Prankl (prankl@acin.tuwien.ac.at)
 * @date 2017
 * @brief
 *
 */

#pragma once

#include <boost/mpl/at.hpp>
#include <boost/mpl/map.hpp>

#include <Eigen/Dense>

#include <pcl/point_types.h>

// Define all point types that include XYZ data
#define V4R_PCL_XYZ_POINT_TYPES (pcl::PointXYZ)(pcl::PointXYZRGB)(pcl::PointNormal)(pcl::PointXYZRGBNormal)

// Define all point types that include RGB data
#define V4R_PCL_RGB_POINT_TYPES (pcl::PointXYZRGB)(pcl::PointXYZRGBNormal)

namespace v4r {

constexpr float NaNf = std::numeric_limits<float>::quiet_NaN();

/**
 * PointXYZRGB
 */
class PointXYZRGB {
 public:
  Eigen::Vector4f pt;
  union {
    struct {
      uint8_t b;
      uint8_t g;
      uint8_t r;
      uint8_t a;
    };
    float rgb;
  };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PointXYZRGB() : pt(Eigen::Vector4f(NaNf, NaNf, NaNf, 1.)) {}

  inline Eigen::Map<Eigen::Vector3f> getVector3fMap() {
    return Eigen::Vector3f::Map(&pt[0]);
  }
  inline const Eigen::Map<const Eigen::Vector3f> getVector3fMap() const {
    return Eigen::Vector3f::Map(&pt[0]);
  }
  inline Eigen::Vector4f &getVector4fMap() {
    return pt;
  }
  inline const Eigen::Vector4f &getVector4fMap() const {
    return pt;
  }

  inline float &operator[](int i) {
    return pt[i];
  }
  inline const float &operator[](int i) const {
    return pt[i];
  }
};

/**
 * PointXYZNormalRGB
 */
class PointXYZNormalRGB {
 public:
  Eigen::Vector4f pt;
  Eigen::Vector4f n;
  union {
    struct {
      uint8_t b;
      uint8_t g;
      uint8_t r;
      uint8_t a;
    };
    float rgb;
  };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PointXYZNormalRGB() : pt(Eigen::Vector4f(NaNf, NaNf, NaNf, 1.)), n(Eigen::Vector4f(NaNf, NaNf, NaNf, 1.)) {}

  inline Eigen::Map<Eigen::Vector3f> getVector3fMap() {
    return Eigen::Vector3f::Map(&pt[0]);
  }
  inline const Eigen::Map<const Eigen::Vector3f> getVector3fMap() const {
    return Eigen::Vector3f::Map(&pt[0]);
  }
  inline Eigen::Vector4f &getVector4fMap() {
    return pt;
  }
  inline const Eigen::Vector4f &getVector4fMap() const {
    return pt;
  }
  inline Eigen::Map<Eigen::Vector3f> getNormalVector3fMap() {
    return Eigen::Vector3f::Map(&n[0]);
  }
  inline const Eigen::Map<const Eigen::Vector3f> getNormalVector3fMap() const {
    return Eigen::Vector3f::Map(&n[0]);
  }
  inline Eigen::Vector4f &getNormalVector4fMap() {
    return n;
  }
  inline const Eigen::Vector4f &getNormalVector4fMap() const {
    return n;
  }

  inline float &operator[](int i) {
    return pt[i];
  }
  inline const float &operator[](int i) const {
    return pt[i];
  }
};

/**
 * PointXYZ (might be wrong!!!!)
 */
class PointXYZ {
 public:
  Eigen::Vector3f pt;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PointXYZ() : pt(Eigen::Vector3f(NaNf, NaNf, NaNf)) {}

  inline Eigen::Vector3f &getVector3fMap() {
    return pt;
  }
  inline const Eigen::Vector3f &getVector3fMap() const {
    return pt;
  }

  inline float &operator[](int i) {
    return pt[i];
  }
  inline const float &operator[](int i) const {
    return pt[i];
  }
};

/// A meta-function that maps a PCL point type to another PCL point type which has the same fields plus normal.
/// Returns the same point type if the input type already has normal field.
///
/// Example usage:
///
/// \code
//  // Assuming PointT is some PCL point type, PointTWithNormal will additionally have normal
/// using PointTWithNormal = typename add_normal<PointT>::type;
/// // Short-hand version of the above
/// using PointTWithNormal = add_normal_t<PointT>;
/// \endcode
template <typename PointT>
struct add_normal {
  typedef boost::mpl::map<boost::mpl::pair<pcl::PointXYZ, pcl::PointNormal>,
                          boost::mpl::pair<pcl::PointNormal, pcl::PointNormal>,
                          boost::mpl::pair<pcl::PointXYZRGB, pcl::PointXYZRGBNormal>,
                          boost::mpl::pair<pcl::PointXYZRGBA, pcl::PointXYZRGBNormal>,
                          boost::mpl::pair<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>,
                          boost::mpl::pair<pcl::PointXYZI, pcl::PointXYZINormal>,
                          boost::mpl::pair<pcl::PointXYZINormal, pcl::PointXYZINormal>>
      PointTypeAssociations;

  BOOST_MPL_ASSERT((boost::mpl::has_key<PointTypeAssociations, PointT>));

  typedef typename boost::mpl::at<PointTypeAssociations, PointT>::type type;
};

template <typename PointT>
using add_normal_t = typename add_normal<PointT>::type;

}  // namespace v4r
