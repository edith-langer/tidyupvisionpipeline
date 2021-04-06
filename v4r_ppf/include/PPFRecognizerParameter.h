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
 * @file PPFRecognizerParameter.h
 * @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
 * @date 2017
 * @brief
 *
 */

#pragma once

#include <fstream>
#include <iostream>
#include <vector>

#include <boost/program_options.hpp>

#include <v4r/apps/CloudSegmenter.h>
#include <v4r/common/intrinsics.h>
#include <v4r/geometry/normals.h>

#include <v4r/io/filesystem.h>

// #include <v4r/keypoints/all_headers.h>
// #include <v4r/ml/all_headers.h>
// #include <v4r/ml/types.h>
// #include <v4r/recognition/global_recognizer.h>
#include <v4r/recognition/hypotheses_verification_param.h>
// #include <v4r/recognition/local_recognition_pipeline.h>
#include <v4r/recognition/multiview_recognizer.h>
#include <ppf_recognition_pipeline.h>
// #include <v4r/segmentation/types.h>

namespace po = boost::program_options;

namespace v4r {

namespace apps {

struct PPFRecognizerParameter {
  const Intrinsics::Ptr cam_;
  bf::path camera_calibration_file_ = v4r::io::getConfigDir() / "rgb_calibration.yaml";
  bf::path rgb_depth_overlap_image_ = v4r::io::getConfigDir() / "rgb_depth_overlap.png";
  std::vector<std::string> object_models_ = {};  ///< object models to load from disk

  PPFRecognitionPipelineParameter ppf_rec_pipeline_;
// MultiviewRecognizerParameter multiview_;

  apps::CloudSegmenterParameter plane_filter_;
  HV_Parameter hv_;

  // pipeline setup
  bool do_ppf_ = false;       ///< enable PPF pipeline

  NormalEstimatorParameter normal_estimator_param_;

  // filter parameter
  double chop_z_ = 3.f;                    ///< Cut-off distance in meter
  float min_height_above_ground_ = 0.01f;  ///< minimum height above ground for input points to be considered (only
                                           ///< used if transform_to_world is set)
  float max_height_above_ground_ = std::numeric_limits<float>::max();  ///< maximum height above ground for input
                                                                       ///< points to be considered (only used if
                                                                       ///< transform_to_world is set)

  bool remove_planes_ =
      true;  ///< if enabled, removes the dominant plane in the input cloud (given thera are at least N
  /// inliers)
  bool remove_non_upright_objects_ =
      false;  ///< removes all objects that are not upright (requires to extract support plane)

  // multi-view parameters
  bool use_multiview_ = false;  ///< if true, transfers verified hypotheses across views
  bool use_multiview_hv_ =
      true;  ///< if true, verifies hypotheses against the registered scene cloud from all input views
  bool use_multiview_with_kp_correspondence_transfer_ =
      false;  ///< if true, transfers keypoints instead of full hypotheses
  ///(see Faeulhammer et al, ICRA 2015)
  bool use_change_detection_ =
      true;  ///< if true, uses change detection to find dynamic elements within observation period
  ///(only for multi-view recognition)
  float tolerance_for_cloud_diff_ = 0.02f;  ///< tolerance in meter for change detection's cloud differencing
  size_t min_points_for_hyp_removal_ =
      50;  ///< how many removed points must overlap hypothesis to be also considered removed
  size_t multiview_max_views_ =
      3;  ///< maximum number of views used for multi-view recognition (if more views are available,
  /// information from oldest views will be ignored)

  size_t icp_iterations_ =
      0;  ///< ICP iterations. Only used if hypotheses are not verified. Otherwise ICP is done inside HV
  double icp_max_corr_dist_ = 0.02;
  double icp_transf_eps_=1e-06;

  bool skip_verification_ = false;     ///< if true, will only generate hypotheses but not verify them

  float max_model_diameter_to_min_plane_ratio_ =
      1.2f;  ///< multiplier for max model diameter used to compute plane removal size threshold

  double min_graph_conf_thr_ = 0.2;
  double min_result_conf_thr_ = 0.75;

  PPFRecognizerParameter() : cam_(new Intrinsics) {}

  void validate();

  /**
   * @brief init parameters
   * @param command_line_arguments (according to Boost program options library)
   * @param section_name section name of program options
   */
  void init(boost::program_options::options_description &desc, const std::string &section_name = "or_multipipeline");
};
}  // namespace apps
}  // namespace v4r
