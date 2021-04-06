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

/**
 * @author Sergey Alexandrov, Thomas Faeulhammer
 * @date 2019
 * @brief
 *
 */

#pragma once

#include <unordered_map>

#include <boost/program_options.hpp>

#include <v4r/common/point_types.h>
#include <ppf/model_search.h>
#include <recognition_pipeline.h>
// #include <v4r/segmentation/all_headers.h>      // @TODO: see if needed

namespace v4r {

struct PPFRecognitionPipelineParameter {
  /// Downsampling resolution for model and scene point clouds (fraction of model diameter).
  /// Note that several other parameters (ppf_distance_quantization_step, pose_clustering_distance_thershold) are
  /// related to this one and might need to be adjusted together.
  float downsampling_resolution_ = 0.05f;

  /// Quantization step for distances in PPF (fraction of model diameter).
  /// Half of the downsampling resolution is a good choice.
  /// See ppf::ModelSearch documentation for details.
  float ppf_distance_quantization_step_ = 0.025f;

  /// Quantization step for angles in PPF (degrees).
  /// See ppf::ModelSearch documentation for details.
  float ppf_angle_quantization_step_ = 5.0f;

  /// Scene subsampling rate.
  /// For each scene point PPF pipeline generates pose hypotheses, which are later clustered. In most cases even
  /// downsampled scene point cloud has too many points and processing all of them results in many redundant pose
  /// hypotheses. The pipeline runtime can be improved (linearly) by processing only a subset of scene points. This
  /// parameter regulates the subsampling rate, i.e. if set to n, only 1 point out of n will be processed.
  size_t scene_subsampling_rate_ = 10;

  /// Number of correspondences (equivalently pose hypotheses) generated per scene point.
  /// If set to one, only the top result of Hough Voting is added to the hypothesis list. Sometimes it happens that
  /// the correct hypothesis is not the top voted one, so it is beneficial to set this parameter above 1. The effect
  /// on runtime is sublinear.
  /// Note that this is the maximum number of generated hypotheses, the actual number can be less if no hypotheses
  /// received enough votes.
  /// See ppf::CorrespondenceFinder documentation for details.
  size_t correspondences_per_scene_point_ = 3;

  /// Minimum number of votes in Hough Voting scheme a candidate correspondence should have to be considered.
  size_t min_votes_ = 3;

  /// Maximum number of pose hypotheses to output.
  /// Internally PPF pipeline generates a large number of hypotheses most of which are invalid. This parameter can be
  /// used to limit the number of output hypotheses to prevent subsequent modules in the recognizer from excessive
  /// processing. Zero means no limit, output all generated pose hypotheses.
  size_t max_hypotheses_ = 3;

  /// Distance threshold for clustering together pose hyotheses (fraction of model diameter).
  /// Double the downsampling resolution is a good choice.
  float pose_clustering_distance_threshold_ = 0.1;

  /// Angular threshold for clustering together pose hyotheses (degrees).
  float pose_clustering_angle_threshold_ = 18.0f;

  /// Use symmetry information, if available.
  /// The symmetry information is used in two ways:
  ///   1) at model search construction phase to reduce the hash size;
  ///   2) at pose clustering phase to group equivalent poses.
  /// The first reduces the model search object size and makes Hough Voting faster. The second has potential to
  /// improve overall results in case of occlusions when the correct hypothesis has similar weight to the incorrect
  /// noise-induced hypotheses.
  bool use_symmetry_ = true;


  /// extend the PPF feature by 2 color vectors in HSV space
  bool use_color_ = false;
  ///this can not be set via the config file for now
  std::vector<float> color_quantization_step_{0.25f, 0.25f, 1.0f};

  ///checks if the color of the scene pair is similar to the color of the model pair before casting a vote in hough space
  /// either use_color_ OR check_col_before_voting_ should be set to true
  bool check_col_before_voting_ =true;

  float inlier_threshold_color_ =
      30.f;  /// allowed chrominance (AB channel of LAB color space) variance for a model point pair to be considered explained
             /// by a object point pair (used when check_col_before_voting_ is set to true

  /// Initialize program options for this parameters object.
  /// \param command_line_arguments (according to Boost program options library)
  /// \param section_name section name of program options
  /// \return unused parameters (given parameters that were not used in this initialization call)
  void init(boost::program_options::options_description &desc, const std::string &section_name = "ppf_pipeline");
};

/// This pipeline implement the approach originally described in:
///
/// Drost, B., Ulrich, M., Navab, N., & Ilic, S. (2010)
/// Model Globally, Match Locally: Efficient and Robust 3D Object Recognition.
/// In Proc. of CVPR
///
/// It also uses some of the extensions proposed in (referred to throughout the documentation as [VLLM18]):
///
/// Vidal, J., Lin, C. Y., Lladó, X., & Martí, R. (2018)
/// A Method for 6D Pose Estimation of Free-form Rigid Objects using Point Pair Features on Range Data.
/// Sensors (Switzerland), 18(8), 1–20
template <typename PointT>
class PPFRecognitionPipeline : public RecognitionPipeline<PointT> {
 private:
  using RecognitionPipeline<PointT>::elapsed_time_;
  using RecognitionPipeline<PointT>::scene_;
  using RecognitionPipeline<PointT>::scene_normals_;
  using RecognitionPipeline<PointT>::obj_hypotheses_;
  using RecognitionPipeline<PointT>::m_db_;
  // using RecognitionPipeline<PointT>::vis_param_;

  using PointTWithNormal = v4r::add_normal_t<PointT>;

  PPFRecognitionPipelineParameter param_;

  void do_recognize(const std::vector<std::string> &model_ids_to_search) override;

  void doInit(const bf::path &trained_dir, bool force_retrain,
              const std::vector<std::string> &object_instances_to_load) override;

  /// Preloaded model search objects for the requested objects types
  std::unordered_map<std::string, ppf::ModelSearch::ConstPtr> model_search_;
  /// Symmetry rotations for requested object types
  /// Each of these rotations aligns object model with itself. Model that have not symmetries will have a single
  /// rotation (identity) in this vector.
  std::unordered_map<std::string, std::vector<Eigen::Matrix3f>> symmetry_rotations_;

 public:
  explicit PPFRecognitionPipeline(const PPFRecognitionPipelineParameter &p = PPFRecognitionPipelineParameter())
  : param_(p) {}

  bool needNormals() const override {
    return true;
  }

  size_t getFeatureType() const override {
    return 0;  // TODO define feature type for PPF
  }

  using Ptr = std::shared_ptr<PPFRecognitionPipeline<PointT>>;
  using ConstPtr = std::shared_ptr<PPFRecognitionPipeline<PointT> const>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace v4r
