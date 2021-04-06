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
 * @file CloudSegmenter.h
 * @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
 * @date 2017
 * @brief
 *
 */

#include <v4r/geometry/normals.h>
#include <v4r/segmentation/all_headers.h>
#include <v4r/segmentation/types.h>
#include <v4r/segmentation/segmenter.h>
#include <v4r/segmentation/plane_extractor.h>
#include <boost/program_options.hpp>

#pragma once

namespace v4r {

namespace apps {

struct CloudSegmenterParameter {
  float chop_z_ = std::numeric_limits<float>::max();  ///< cut-off distance in meter. Points further away than this
                                                      ///< threshold will be neglected
  size_t min_plane_inliers_ = 0;                      ///< minimum number of inlier points for a plane to be valid
  float min_plane_diameter_ = 1.f;                    ///< minimum plane diameter in meter for a plane to be valid
  bool skip_segmentation_ = false;                    ///< if true, skips segmentation
  bool skip_plane_extraction_ = false;                ///< if true, skips plane extraction
  bool remove_planes_ =
      false;  ///< if true, removes plane from input cloud only. If false, removes plane and everything below
              /// it (i.e. further away from the camera)
  bool remove_selected_plane_ =
      true;  ///< if true, removes the selected plane (either dominant or the one parallel and higher)
  bool remove_points_below_selected_plane_ =
      true;  ///< if true, removes any point not above selected plane (plane gets either
             /// selected by dominant (=plane with most inliers) or by the highest plane
  /// parallel to this dominant one)
  bool use_highest_plane_ = false;  ///< if true, selects the highest plane parallel to the dominant plane
  float cosinus_angle_for_planes_to_be_parallel_ =
      0.95f;  ///< the minimum cosinus angle of the surface normals of two planes
              /// such that the two planes are considered parallel (only used if
  /// check for higher plane is enabled)
  float min_distance_to_plane_ = 0.f;  ///< minimum distance in meter a point needs to have to be considered above

  SegmentationType segmentation_method_ = v4r::SegmentationType::ORGANIZED_CONNECTED_COMPONENTS;
  PlaneExtractionType plane_extraction_method_ = v4r::PlaneExtractionType::TILE;
  NormalEstimatorParameter normal_computation_parameter_;
  PlaneExtractorParameter plane_extractor_parameter_;
  SegmenterParameter segmentation_parameter_;

  /**
   * @brief init parameters
   * @param command_line_arguments (according to Boost program options library)
   * @param section_name section name of program options
   */
  void init(boost::program_options::options_description &desc, const std::string &section_name = "segmentation");
};

/**
 * segmentation algorithm
 * @author Thomas Faeulhammer
 */

template <typename PointT>
class CloudSegmenter {
 private:
  typename v4r::PlaneExtractor<PointT>::Ptr plane_extractor_;
  typename v4r::Segmenter<PointT>::Ptr segmenter_;
  std::vector<std::vector<int>> found_clusters_;
  pcl::PointCloud<pcl::Normal>::ConstPtr normals_;
  std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> planes_;
  std::vector<std::vector<int>> plane_inliers_;
  typename pcl::PointCloud<PointT>::Ptr processed_cloud_;
  Eigen::Vector4f selected_plane_;

  CloudSegmenterParameter param_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CloudSegmenter(const CloudSegmenterParameter &p = CloudSegmenterParameter()) : param_(p) {}

  /**
   * @brief initialize initialize Point Cloud Segmenter (sets up plane extraction, segmentation and potential normal
   * estimator)
   */
  void initialize();

  /**
   * @brief recognize recognize objects in point cloud
   * @param cloud (organized) point cloud
   * @param transform_to_world rigid transform that aligns points into a world reference frame (which is assumed to have
   * z=0 aligned to floor). This parameter is used to filter points based on height above ground.
   * @return
   */
  void segment(const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
               const boost::optional<Eigen::Matrix4f> &transform_to_world = boost::none);

  /**
   * @brief getClusters
   * @param cluster_indices
   */
  std::vector<std::vector<int>> getClusters() const {
    return found_clusters_;
  }

  /**
   * @brief getPlanes
   * @return extracted planar surfaces
   */
  std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> getPlanes() const {
    return planes_;
  }

  /**
   * @brief setNormals sets the normals associated to the input cloud (if not set, they will be computed inside if
   * necessary)
   * @param normals
   */
  void setNormals(const typename pcl::PointCloud<pcl::Normal>::ConstPtr &normals) {
    normals_ = normals;
  }

  /**
   * @brief getPlaneInliers indices of the cloud that belong to a plane with given inlier threshold parameter (or as
   * specified by the plane extraction method)
   * @return plane inliers
   */
  std::vector<std::vector<int>> getPlaneInliers() const {
    return plane_inliers_;
  }

  /**
   * @brief getProcessedCloud
   * @return processed/filtered cloud
   */
  typename pcl::PointCloud<PointT>::Ptr getProcessedCloud() const {
    return processed_cloud_;
  }

  pcl::PointCloud<pcl::Normal>::ConstPtr getNormals() const {
    return normals_;
  }

  /**
   * @brief getPlane
   * @return dominant plane or highest plane parallel to dominant plane (depending on chosen parameter)
   */
  Eigen::Vector4f getSelectedPlane() const {
    return selected_plane_;
  }

  typedef std::shared_ptr<CloudSegmenter> Ptr;
  typedef std::shared_ptr<CloudSegmenter const> ConstPtr;
};
}  // namespace apps
}  // namespace v4r
