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
 * @file  segmenter.h
 * @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
 * @date   April, 2016
 * @brief Base class for segmentation
 *
 */

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <boost/program_options.hpp>

namespace v4r {

struct SegmenterParameter {
 public:
  size_t min_cluster_size_ = 500;                              ///< minimum number of points in a cluster
  size_t max_cluster_size_ = std::numeric_limits<int>::max();  ///< minimum number of points in a cluster
  double distance_threshold_ =
      0.01;  ///< tolerance in meters for difference in perpendicular distance (d component of plane
             /// equation) to the plane between neighboring points, to be considered part of the same
  /// plane
  // 0.035f; for organized connected componeents
  double angular_threshold_deg_ =
      10.;  ///< tolerance in gradients for difference in normal direction between neighboring
            /// points, to be considered part of the same plane
  int wsize_ = 5;
  bool z_adaptive_ = true;  ///< if true, scales the smooth segmentation parameters linear with distance (constant till
                            ///< 1m at the given parameters)

  // parameters for smooth clustering
  bool compute_planar_patches_only_ = false;  ///< if true, only compute planar surface patches
  float planar_inlier_dist_ = 0.02f;          ///< maximum allowed distance of a point to the plane
  float octree_resolution_ = 0.01f;           ///< octree resolution in meter
  bool force_unorganized_ = false;  ///< if true, searches for neighboring points using a kdtree and not exploiting the
                                    ///< organized pixel structure (even if input cloud is organized)
  float curvature_threshold_ = 0.04f;                       ///< smooth clustering threshold for curvature
  bool downsample_before_segmentation_ = false;             ///< downsample the scene before euclidean clustering
                                                            ///< is applied
  float downsample_before_segmentation_resolution_ = 0.01;  ///< the resolution in meter used for
                                                            ///< uniform sampling before euclidean clustering

  /**
   * @brief init parameters
   * @param command_line_arguments (according to Boost program options library)
   * @param section_name section name of program options
   */
  void init(boost::program_options::options_description &desc, const std::string &section_name = "segmenter");
};

template <typename PointT>
class Segmenter {
 protected:
  typename pcl::PointCloud<PointT>::ConstPtr scene_;  ///< point cloud to be segmented
  pcl::PointCloud<pcl::Normal>::ConstPtr normals_;    ///< normals of the cloud to be segmented
  std::vector<std::vector<int>>
      clusters_;  ///< segmented clusters. Each cluster represents a bunch of indices of the input cloud

 public:
  virtual ~Segmenter() = default;

  Segmenter() = default;

  /**
   * @brief sets the cloud which ought to be segmented
   * @param cloud
   */
  void setInputCloud(const typename pcl::PointCloud<PointT>::ConstPtr &cloud) {
    scene_ = cloud;
  }

  /**
   * @brief sets the normals of the cloud which ought to be segmented
   * @param normals
   */
  void setNormalsCloud(const pcl::PointCloud<pcl::Normal>::ConstPtr &normals) {
    normals_ = normals;
  }

  /**
   * @brief get segmented indices
   * @param indices
   */
  void getSegmentIndices(std::vector<std::vector<int>> &indices) const {
    indices = clusters_;
  }

  virtual bool getRequiresNormals() const = 0;

  /**
   * @brief segment
   */
  virtual void segment() = 0;

  typedef std::shared_ptr<Segmenter<PointT>> Ptr;
  typedef std::shared_ptr<Segmenter<PointT> const> ConstPtr;
};
}  // namespace v4r
