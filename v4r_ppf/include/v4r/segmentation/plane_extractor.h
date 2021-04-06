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
 * @file plane_extractor.h
 * @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
 * @date 2017
 * @brief
 *
 */

#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <boost/optional.hpp>
#include <boost/program_options.hpp>

namespace v4r {

struct PlaneExtractorParameter {
  size_t max_iterations_ = 100;          ///< maximum number of iterations the sample consensus method will run
  size_t min_num_plane_inliers_ = 1000;  ///< minimum number of plane inliers
  double distance_threshold_ =
      0.01;  ///< tolerance in meters for difference in perpendicular distance (d component of plane
             /// equation) to the plane between neighboring points, to be considered part of the same
  /// plane
  bool compute_all_planes_ = true;    ///< if true, computes all planes (also if method does not compute all of them
                                      /// intrinsically)
  bool optimize_cofficients_ = true;  ///< true for enabling model coefficient refinement, false otherwise
  double eps_angle_ =
      M_PI * 10. /
      180.;  ///< maximum alloweSAmple Consensus (SAC)d difference between the plane normal and the given axis.
  int model_type_ = pcl::SACMODEL_PLANE;  ///< Model type used for SAmple Consensus (SAC)
  int method_type_ = pcl::SAC_RANSAC;     ///< Method type for SAmple Consensus (SAC)
  double probability_ = 0.99;
  double samples_max_distance_ = 0.;  ///< maximum distance in meter allowed when drawing random samples
  bool z_adaptive_ = false;           ///< whether to scale the threshold based on range from the sensor (for organized
                                      ///< multiplane detection only)
  bool check_normals_ = false;  ///< if true, discards points that are on the plane normal but have a surface normal
                                ///< orientation that is different to the plane's surface normal orientation by
                                ///< eps_angle_

  // Parameters for Tile
  size_t minNrPatches_ = 5;  ///< The minimum number of blocks that are allowed to spawn a plane
  size_t patchDim_ = 10;     ///< Patches are made of pixel squares that have exactly these side length
  float minBlockInlierRatio_ =
      0.95f;  ///< The minimum ratio of points that have to be in a patch before it would get discarded.
  bool pointwiseNormalCheck_ =
      false;  ///< Activating this allowes to reduce a lot of calculations and improve speed by a lot
  bool useVariableThresholds_ = true;  ///< useVariableThresholds
  float maxInlierBlockDist_ = 0.005f;  ///< The maximum distance two adjacent patches are allowed to be out of plane
  bool doZTest_ = true;                ///< Only the closest possible points get added to a plane

  /**
   * @brief init parameters
   * @param desc boost program description object(according to Boost program options library)
   * @param section_name section name of program options
   */
  void init(boost::program_options::options_description &desc, const std::string &section_name = "plane_extractor");
};

/**
 * @brief The PlaneExtractor class is an abstract class which extracts planar surfaces from a point cloud
 */
template <typename PointT>
class PlaneExtractor {
 protected:
  typename pcl::PointCloud<PointT>::ConstPtr cloud_;     ///< input cloud
  pcl::PointCloud<pcl::Normal>::ConstPtr normal_cloud_;  ///< surface normals associated to input cloud
  std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>>
      all_planes_;  ///< all extracted planes (if segmentation algorithm supports it)
  PlaneExtractorParameter param_;
  std::vector<std::vector<int>> plane_inliers_;

  virtual void do_compute(const boost::optional<const Eigen::Vector3f> &search_axis) = 0;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PlaneExtractor(const PlaneExtractorParameter &p = PlaneExtractorParameter()) : param_(p) {}

  /**
   * @brief compute extract planes
   * @param axis the axis along which we need to search for a plane parallel/perpendicular to (used for certain model
   types only)
   */
  void compute(const boost::optional<const Eigen::Vector3f> &search_axis = boost::none);

  /**
   * @brief getRequiresNormals
   * @return true if method requires normal cloud to be set
   */
  virtual bool getRequiresNormals() const = 0;

  /**
   * @brief sets the cloud which ought to be segmented
   * @param cloud
   */
  void setInputCloud(const typename pcl::PointCloud<PointT>::ConstPtr &cloud) {
    cloud_ = cloud;
  }

  /**
   * @brief sets the normals of the cloud which ought to be segmented
   * @param normals
   */
  void setNormalsCloud(const pcl::PointCloud<pcl::Normal>::ConstPtr &normals) {
    normal_cloud_ = normals;
  }

  /**
   * @brief getPlanes
   * @return all extracted planes
   */
  std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> getPlanes() const {
    return all_planes_;
  }

  /**
   * @brief getPlaneInliers
   * @return indices of cloud corresponding to the inliers for each plane
   */
  std::vector<std::vector<int>> getPlaneInliers();

  typedef std::shared_ptr<PlaneExtractor<PointT>> Ptr;
  typedef std::shared_ptr<PlaneExtractor<PointT> const> ConstPtr;
};
}  // namespace v4r
