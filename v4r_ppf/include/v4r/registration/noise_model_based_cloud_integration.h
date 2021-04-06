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
 * @file noise_model_based_cloud_integration.h
 * @author Aitor Aldoma (aldoma@acin.tuwien.ac.at), Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
 * @date 2013
 * @brief
 *
 */

#pragma once

#include <pcl/point_cloud.h>
#include <v4r/common/intrinsics.h>
#include <boost/optional.hpp>
#include <boost/program_options.hpp>

namespace v4r {

struct NMBasedCloudIntegrationParameter {
  size_t min_points_per_voxel_ = 2;   ///< the minimum number of points in a leaf of the octree of the big cloud. If
                                      ///< adaptive threshold is set, this will be the lower bound of the adaptive
                                      ///< threshold.
  float octree_resolution_ = 0.003f;  ///< resolution of the octree of the big point cloud
  bool average_ =
      false;  ///< if true, takes the average color (for each color componenent) and normal within all the points in
              /// the leaf of the octree. Otherwise, it takes the point within the octree with the best noise weight
  float min_px_distance_to_depth_discontinuity_ =
      3.f;  ///< points of the input cloud within this distance (in pixel) to its
            /// closest depth discontinuity pixel will be removed. Uses PCL's organized edge detection algorithm to
            /// compute distance of each pixel
            //             /// to these discontinuites.
  float viewpoint_surface_orienation_dotp_thresh_ = 0.6;  ///< threshold for the inner product of viewpoint and surface
  ///< normal orientation to have more importance than the
  ///< weight variable
  float px_distance_to_depth_discontinuity_thresh_ =
      3.f;  ///< threshold up to what point the distance to depth discontinuity is more important than other metrics
  bool resolution_adaptive_min_points_ = true;  ///< if true, sets the requirements for the minimum number of points to
                                                ///< a certain percentage of the maximum number of points within a voxel
  float adaptive_min_points_percentage_thresh_ =
      0.1f;  ///< if resolution_adaptive_min_points_ is set to true, this parameter will set the minimum number of
             ///< points needed within each voxel as percentage of the maximum number of points within each voxel. If
             ///< this threshold is below the parameter min_points_per_voxel_, it will be set to min_points_per_voxel_.
  bool use_nguyen_ = true;  ///< if true, uses Nguyens noise model (Nguyen et al., 3DIMPVT 2012.)

  /**
   * @brief init parameters
   * @param command_line_arguments (according to Boost program options library)
   * @param section_name section name of program options
   */
  void init(boost::program_options::options_description &desc,
            const std::string &section_name = "nm_based_cloud_integration");
};

/**
 * @brief reconstructs a point cloud from several input clouds. Each point of the input cloud is associated with a
 * weight
 * which states the measurement confidence ( 0... max noise level, 1... very confident). Each point is accumulated into
 * a
 * big cloud and then reprojected into the various image planes of the input clouds to check for conflicting points.
 * Conflicting points will be removed and the remaining points put into an octree
 * @author Thomas Faeulhammer, Aitor Aldoma
 * @date December 2015
 */
template <class PointT>
class NMBasedCloudIntegration {
 private:
  NMBasedCloudIntegrationParameter param_;
  v4r::Intrinsics::ConstPtr cam_;

  class PointInfo {
   private:
    size_t num_pts_for_average_;  ///< number of points used for building average

   public:
    PointT pt_;
    pcl::Normal normal_;
    float distance_to_depth_discontinuity_;
    float weight_;
    float dotp_;  ///< inner product of viewray and surface normal

    PointInfo() : num_pts_for_average_(1), weight_(std::numeric_limits<float>::max()) {
      pt_.getVector3fMap() = Eigen::Vector3f::Zero();
      pt_.r = pt_.g = pt_.b = 0.f;
      normal_.getNormalVector3fMap() = Eigen::Vector3f::Zero();
      normal_.curvature = 0.f;
    }

    void moving_average(const PointInfo &new_pt) {
      num_pts_for_average_++;

      double w_old = static_cast<double>(num_pts_for_average_) / (num_pts_for_average_ + 1.f);
      double w_new = 1.f / (static_cast<double>(num_pts_for_average_) + 1.f);

      pt_.getVector3fMap() = w_old * pt_.getVector3fMap() + w_new * new_pt.pt_.getVector3fMap();
      pt_.r = w_old * pt_.r + w_new * new_pt.pt_.r;
      pt_.g = w_old * pt_.g + w_new * new_pt.pt_.g;
      pt_.b = w_old * pt_.b + w_new * new_pt.pt_.b;

      Eigen::Vector3f new_normal = new_pt.normal_.getNormalVector3fMap();
      new_normal.normalize();
      normal_.getNormalVector3fMap() = w_old * normal_.getNormalVector3fMap() + w_new * new_normal;
      normal_.curvature = static_cast<float>(w_old * normal_.curvature + w_new * new_pt.normal_.curvature);
    }
  };

  std::vector<PointInfo> big_cloud_info_;
  pcl::PointCloud<pcl::Normal>::Ptr output_normals_;

  void cleanUp() {
    big_cloud_info_.clear();
  }

 public:
  NMBasedCloudIntegration(
      const NMBasedCloudIntegrationParameter &p = NMBasedCloudIntegrationParameter(),
      const v4r::Intrinsics::ConstPtr &cam = std::make_shared<v4r::Intrinsics>(v4r::Intrinsics::PrimeSense()))
  : param_(p), cam_(cam) {}

  /**
   * @brief getOutputNormals
   * @return Normals of the registered cloud
   */
  pcl::PointCloud<pcl::Normal>::Ptr getOutputNormals() const {
    return output_normals_;
  }

  /**
   * @brief add a new view to the noise model based cloud integration
   * @param cloud organized input cloud in the camera reference frame
   * @param normals associated surface normals
   * @param view_id view id to allow to get view origin of all registered points
   * @param transform_to_global_reference_frame SE(3) camera pose bringing the input cloud into a common reference frame
   * with the other clouds
   */
  void addView(const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
               const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
               const Eigen::Matrix4f &transform_to_global_reference_frame,
               const boost::optional<std::vector<int>> &indices = boost::none);

  /**
   * @brief compute the registered point cloud taking into account the noise model of the cameras
   * @param registered cloud
   */
  void compute(typename pcl::PointCloud<PointT>::Ptr &output);
};
}  // namespace v4r
