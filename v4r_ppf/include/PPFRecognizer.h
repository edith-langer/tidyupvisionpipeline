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
 * @file PPFRecognizer.h
 * @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
 * @date 2017
 * @brief
 *
 */

#pragma once

#include <boost/serialization/vector.hpp>

#include <PPFRecognizerParameter.h>
#include <v4r/common/point_types.h>
#include <v4r/geometry/normals.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/hypotheses_verification.h>

namespace bf = boost::filesystem;

namespace v4r {

namespace apps {

/**
 * @brief Class that sets up multi-pipeline object recognizer
 * @author Thomas Faeulhammer
 * @tparam PointT
 */
template <typename PointT>
class PPFRecognizer {
 private:
  using PointTWithNormal = v4r::add_normal_t<PointT>;

  typename v4r::RecognitionPipeline<PointT>::Ptr mrec_;                             ///< multi-pipeline recognizer
  typename v4r::HypothesisVerification<PointT>::Ptr hv_;                            ///< hypothesis verification object

  bf::path models_dir_;

  PPFRecognizerParameter param_;  ///< parameters for object recognition

  typename Source<PointT>::Ptr model_database_;  ///< object model database

  void validate();  ///< checks input data and paramer

  /**
   * @brief helper class for multi-view recognition system
   */
  class View {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typename pcl::PointCloud<PointT>::ConstPtr cloud_;
    typename pcl::PointCloud<PointT>::Ptr processed_cloud_;
    typename pcl::PointCloud<PointT>::Ptr removed_points_;
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_;
    Eigen::Matrix4f camera_pose_;
  };
  std::vector<View> views_;  ///< all views in sequence


  typename pcl::PointCloud<PointT>::Ptr registered_scene_cloud_;  ///< registered point cloud of all processed input
                                                                  /// clouds in common camera reference frame

  std::vector<std::pair<std::string, float>>
      elapsed_time_;  ///< measurements of computation times for various components

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PPFRecognizer(const PPFRecognizerParameter &p = PPFRecognizerParameter()) : param_(p) {}

  /**
   * @brief initialize initialize Object recognizer (sets up model database, recognition pipeline and hypotheses
   * verification)
   * @param force_retrain if set, will force to retrain object recognizer even if trained data already exists
   */
  void setup(bool force_retrain = false, bool use_color = false);

  /**
   * @brief recognize recognize objects in point cloud together with their 6DoF pose
   * @param cloud (organized) point cloud
   * @param obj_models_to_search object model identities to detect. If empty, all object models in database will be
   * tried to detect
   * @param transform_to_world rigid transform that aligns points into a world reference frame (which is assumed to have
   * z=0 aligned to floor). This parameter is used to filter points based on height above ground.
   * @param region_of_interest defines the region of interest either as a binary mask (0/>0 out-/insideROI) or a set of
   * pixel indices which belong to the ROI. If empty, ROI will be defined as the whole input space.
   * @return detected objects (accepted and rejected(!) object hypotheses)
   */
  std::vector<ObjectHypothesesGroup> recognize(const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
      const std::vector<std::string> &obj_models_to_search = std::vector<std::string>(),
      const boost::optional<pcl::PointCloud<pcl::Normal>::Ptr> &cloud_normals = boost::none,
      const boost::optional<Eigen::Matrix4f> &transform_to_world = boost::none,
      cv::InputArray &region_of_interest = cv::noArray());

  /**
   * @brief get point cloud of object model
   * @param model_name identity of object model to return
   * @param resolution_mm  resolution of point cloud model in milli meter
   * @return point cloud of object model or nullptr if it does not exist
   */
  typename pcl::PointCloud<PointTWithNormal>::ConstPtr getModel(const std::string &model_name,
                                                                int resolution_mm=0) const {
    const auto mdb = mrec_->getModelDatabase();
    const auto model = mdb->getModelById("", model_name);
    if (!model) {
      std::cerr << "Could not find model with name " << model_name << std::endl;
      return nullptr;
    }

    if (resolution_mm > 0)
        return model->getAssembled(DownsamplerParameter(resolution_mm / 1000.f));
    else
        return model->getAssembled();
  }

  /**
   * @brief get path to models directory
   * @return
   */
  bf::path getModelsDir() const {
    return models_dir_;
  }

  /**
   * @brief set path to models directory
   * @param dir
   */
  void setModelsDir(const bf::path &dir) {
    models_dir_ = dir;
  }

  /**
   * @brief getElapsedTimes
   * @return compuation time measurements for various components
   */
  std::vector<std::pair<std::string, float>> getElapsedTimes() const {
    return elapsed_time_;
  }

  /**
   * @brief getParam get recognition parameter
   * @return parameter
   */
  PPFRecognizerParameter getParam() const {
    return param_;
  }

  /**
   * @brief getCamera get camera intrinsic parameters
   * @return camera intrinsic parameter
   */
  Intrinsics getCameraIntrinsics() const {
    return *param_.cam_;
  }

  /**
   * @brief setCamera set the camera intrinsics used for z-buffering
   * @param cam camera intrinsic parameters
   */
  void setCameraIntrinsics(const Intrinsics &cam) {
    *param_.cam_ = cam;
  }

  /**
   * @brief resetMultiView resets all state variables of the multi-view and initializes a new multi-view sequence
   */
  void resetMultiView();
};
}  // namespace apps
}  // namespace v4r
