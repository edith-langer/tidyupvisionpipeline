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
 * @file Clustering.h
 * @author Johann Prankl (prankl@acin.tuwien.ac.at), Thomas Faeulhammer
 * @date 2017
 * @brief
 *
 */
#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <v4r/common/impl/DataMatrix2D.hpp>
#include <vector>

namespace v4r {

/**
 * @brief the Cluster class represents the properties of a point cloud segment
 * @todo merge custom and PCL point cloud data type
 */
class Cluster {
  bool table_plane_set_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  std::vector<int> indices_;              ///< segmented cloud to be recognized (if empty, all points will be processed)
  Eigen::Vector4f table_plane_;           ///< extracted table plane of input cloud (if used for pose estimation)
  Eigen::Vector4f centroid_;              ///< centroid of cluster
  Eigen::Vector3f elongation_;            ///< elongations along the principal component of cluster
  Eigen::Matrix4f eigen_pose_alignment_;  ///< transform which aligns cluster with coordinate frame such that cluster
                                          ///< centroid aligns with origin and principal axis with coordintate frames
  Eigen::Vector3f eigen_values_;          ///< Eigen values for each Eigen vector
  Eigen::Matrix3f eigen_vectors_;         ///< Eigen values for each Eigen vector

  /**
   * @brief Cluster represents a segment from a point cloud
   * @param cloud full point cloud
   * @param indices indices that belong to the segment
   * @param compute_properties if true, computes centroid, principal components and transformation to align with
   * principal axis
   */
  template <typename PointT>
  Cluster(const pcl::PointCloud<PointT> &cloud, const std::vector<int> &indices = std::vector<int>(),
          bool compute_properties = true);

  /**
   * @brief setTablePlane
   * @param table_plane
   */
  void setTablePlane(const Eigen::Vector4f &table_plane) {
    table_plane_ = table_plane;
    table_plane_set_ = true;
  }

  bool isTablePlaneSet() const {
    return table_plane_set_;
  }

  float sqr_sigma = 0.f;
  Eigen::VectorXf data;

  Cluster() = delete;
  Cluster(const Eigen::VectorXf &d) : sqr_sigma(0.f), data(d) {}
  Cluster(const Eigen::VectorXf &d, int idx) : sqr_sigma(0.f), data(d) {
    indices_.push_back(idx);
  }

  typedef std::shared_ptr<Cluster> Ptr;
  typedef std::shared_ptr<Cluster const> ConstPtr;
};

/**
 * Clustering
 */
class Clustering {
 protected:
  std::vector<Cluster::Ptr> clusters;

 public:
  Clustering() {}
  virtual ~Clustering() {}

  virtual void cluster(const DataMatrix2Df &) = 0;
  virtual void getClusters(std::vector<std::vector<int>> &) = 0;
  virtual void getCenters(DataMatrix2Df &) = 0;
};
}  // namespace v4r
