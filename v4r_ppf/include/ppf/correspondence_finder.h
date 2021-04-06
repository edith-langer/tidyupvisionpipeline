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

#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/organized.h>

#include <ppf/correspondence.h>
#include <ppf/model_search.h>

namespace ppf {

Eigen::Vector3f float2lab(float col) {
    //from float to RGB
    Eigen::Vector3i rgb;
    int32_t col_int = reinterpret_cast<int32_t&>(col);
    rgb[0] = (col_int >> 16) & 0x0000ff;
    rgb[1] = (col_int >> 8)  & 0x0000ff;
    rgb[2] = (col_int)     & 0x0000ff;

    cv::Mat rgb_cv (1,1, CV_8UC3);  //this has some information loss because Lab values are also just uchar and not float
    rgb_cv.at<cv::Vec3b>(0,0)[0] = rgb[0];
    rgb_cv.at<cv::Vec3b>(0,0)[1] = rgb[1];
    rgb_cv.at<cv::Vec3b>(0,0)[2] = rgb[2];
    cv::Mat lab_cv;
    cv::cvtColor(rgb_cv, lab_cv, CV_RGB2Lab);

    Eigen::Vector3f lab; //from opencv lab to orig lab definition
    lab << lab_cv.at<cv::Vec3b>(0,0)[0] / 2.55f, lab_cv.at<cv::Vec3b>(0,0)[1] - 128.f, lab_cv.at<cv::Vec3b>(0,0)[2] - 128.f;
    return lab;
}


/// Given a scene point cloud and a PPF model search object, this class finds correspondences between scene points and
/// model points.
///
/// The correspondences are queried using the find() method, which accepts an index into the scene point cloud. Under
/// the assumption that the specified point actually belongs to the model, this class finds to which particular point(s)
/// on the model it corresponds. This is done by analyzing the point neighborhood and using the PPFModelSearch object to
/// look up similar point pairs, which then cast votes in the Hough Voting space. See [VLLM18], section 2.2.4 for
/// further details.
///
/// The computed correspondence have a weight (number of votes) and a pose attached. The pose describes how the model
/// has to be transformed such that it is aligned with the scene point cloud.
///
/// The number of correspondences output can be controlled with the setMaxCorrespondences() method. Allowing multiple
/// correspondences adds robustness since the top-voted correspondence will not necessarily be the correct one. On the
/// other hand, this increases the runtime of the find() procedure and makes correspondence grouping further down the
/// pipeline more complex.
///
/// The user has to provide scene point cloud and ModelSearch object that describes the model before running queries.
template <typename PointT>
class CorrespondenceFinder {
  static_assert(pcl::traits::has_normal<PointT>::value, "Template point type should have normal field");

 public:
  using PointCloud = pcl::PointCloud<PointT>;
  using PointCloudPtr = typename PointCloud::Ptr;
  using PointCloudConstPtr = typename PointCloud::ConstPtr;

  /// Set scene point cloud.
  void setInput(const PointCloudConstPtr& input) {
    scene_ = input;
    if (scene_->isOrganized())
      scene_search_tree_.reset(new pcl::search::OrganizedNeighbor<PointT>);
    else
      scene_search_tree_.reset(new pcl::search::KdTree<PointT>);
    scene_search_tree_->setInputCloud(scene_);
  }

  /// Set search object for querying model local coordinates of point pairs.
  void setModelSearch(const ModelSearch::ConstPtr& model_search) {
    model_search_ = model_search;
    const std::vector<float>model_point_colors = model_search_->getModelPointColors();
    if (model_point_colors.size() != 0) {
        model_lab_cols_.resize(model_point_colors.size());
        for (size_t i = 0; i < model_point_colors.size(); i++) {
            model_lab_cols_[i] = float2lab(model_point_colors[i]);
        }
    }
  }

  /// Find correspondences for a scene point with a given index.
  Correspondence::Vector find(uint32_t scene_index, ModelSearch::FeatureType ppf_type) const;

  /// Set the maximum number of correspondences that find() is allowed to output.
  inline void setMaxCorrespondences(size_t max_correspondences) {
    max_correspondences_ = max_correspondences;
  }

  /// Set the minimum number of votes in Hough Voting scheme a candidate correspondence should have to be output.
  inline void setMinVotes(size_t min_votes) {
    min_votes_ = min_votes;
  }

  /// Set number of bins for angular dimension of the Hough Voting space.
  /// Default value is 30, which corresponds to 12 degrees discretization and seems to me a reasonable choice.
  inline void setNumAngularBins(size_t num_angular_bins) {
    num_angular_bins_ = num_angular_bins;
  }

  //Should the color of the scene and model indices be checked for similarity before casting a vote?
  inline void setUseColorCheck(bool check_color, float color_inlier_thr) {
      if (!pcl::traits::has_color<PointT>::value && check_color) {
          LOG(WARNING) << "Object point cloud does not contain color information. Can not check for color similarity before casting a vote!";
          check_col_before_voting_ = false;
      } else {
        check_col_before_voting_ = check_color;
        color_inlier_thr_ = color_inlier_thr_;
      }
  }

 private:
  PointCloudConstPtr scene_;
  typename pcl::search::Search<PointT>::Ptr scene_search_tree_;
  ModelSearch::ConstPtr model_search_;
  size_t max_correspondences_ = 1;
  size_t min_votes_ = 3;
  size_t num_angular_bins_ = 30;
  std::vector<Eigen::Vector3f> model_lab_cols_;
  bool check_col_before_voting_ = false;
  float color_inlier_thr_ = 30;
};

}  // namespace ppf
