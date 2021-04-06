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
 * @file hypotheses_verification.h
 * @author Aitor Aldoma (aldoma@acin.tuwien.ac.at), Federico Tombari, Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
 * @date 2017
 * @brief
 *
 */

#pragma once

#include <v4r/common/color_comparison.h>
#include <v4r/common/intrinsics.h>
#include <v4r/common/point_types.h>
#include <v4r/geometry/normals.h>
#include <v4r/recognition/hypotheses_verification_param.h>
#include <v4r/recognition/hypotheses_verification_visualization.h>
#include <v4r/recognition/object_hypothesis.h>
#include <v4r/recognition/outline_verification.h>

#include <pcl/common/angles.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/octree/octree.h>
#include <pcl/registration/icp.h>
#include <pcl/search/kdtree.h>

#include <pcl/octree/octree_pointcloud_pointvector.h>
#include <pcl/octree/impl/octree_iterator.hpp>

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/mpl/at.hpp>
#include <boost/mpl/map.hpp>

namespace v4r {

// forward declarations
template <typename PointT>
class HV_ModelVisualizer;
template <typename PointT>
class HV_CuesVisualizer;
template <typename PointT>
class HV_PairwiseVisualizer;

class PtFitness {
 public:
  float fit_;
  size_t rm_id_;

  bool operator<(const PtFitness &other) const {
    return this->fit_ < other.fit_;
  }

  PtFitness(float fit, size_t rm_id) : fit_(fit), rm_id_(rm_id) {}
};

/**
 * \brief A hypothesis verification method for 3D Object Instance Recognition
 * \author Thomas Faeulhammer (based on the work of Federico Tombari and Aitor Aldoma)
 * \date April, 2016
 */
template <typename PointT>
class HypothesisVerification {
  friend class HV_ModelVisualizer<PointT>;
  friend class HV_CuesVisualizer<PointT>;
  friend class HV_PairwiseVisualizer<PointT>;

  HV_Parameter param_;

 public:
  typedef std::shared_ptr<HypothesisVerification<PointT>> Ptr;
  typedef std::shared_ptr<HypothesisVerification<PointT> const> ConstPtr;

 protected:
  using PointTWithNormal = v4r::add_normal_t<PointT>;

  mutable typename HV_ModelVisualizer<PointT>::Ptr vis_model_;
  mutable typename HV_CuesVisualizer<PointT>::Ptr vis_cues_;
  mutable typename HV_PairwiseVisualizer<PointT>::Ptr vis_pairwise_;

  Intrinsics::ConstPtr
      cam_;  ///< rgb camera intrinsics. Used for projection of the object hypotheses onto the image plane.

  bool visualize_pairwise_cues_ = false;  ///< visualizes the pairwise cues. Useful for debugging

  typename Source<PointT>::ConstPtr m_db_;  ///< model data base

  typename pcl::PointCloud<PointTWithNormal>::ConstPtr scene_w_normals_;          ///< scene cloud (with normals)
  typename pcl::PointCloud<PointTWithNormal>::ConstPtr scene_cloud_downsampled_;  ///< Downsampled scene point cloud
  std::shared_ptr<std::vector<int>> scene_sampled_indices_;                       ///< downsampled indices of the scene
  Eigen::VectorXi scene_indices_map_;  ///< saves relationship between indices of the input cloud and indices of the
                                       ///< downsampled input cloud

  std::vector<std::vector<typename HVRecognitionModel<PointT>::Ptr>> obj_hypotheses_groups_;
  std::vector<typename HVRecognitionModel<PointT>::Ptr>
      global_hypotheses_;  ///< all hypotheses not rejected by individual verification

  std::map<std::string, std::shared_ptr<pcl::octree::OctreePointCloudPointVector<PointTWithNormal>>>
      octree_model_representation_;  ///< for each model we create an octree representation (used for computing visible
                                     /// points)

  float Lmin_ = 0.f, Lmax_ = 100.f;
  int bins_ = 50;

  Eigen::MatrixXf intersection_cost_;      ///< represents the pairwise intersection cost
  Eigen::MatrixXi smooth_region_overlap_;  ///< represents if two hypotheses explain the same smooth region
  // cached variables for speed-up
  float eps_angle_threshold_rad_;

  std::vector<std::vector<PtFitness>> scene_pts_explained_solution_;

  static std::vector<std::pair<std::string, float>>
      elapsed_time_;  ///< measurements of computation times for various components

  struct Solution {
    boost::dynamic_bitset<> solution_;
    double cost_;
  } best_solution_;  ///< costs for each possible solution

  std::set<size_t> evaluated_solutions_;
  void search();

  std::vector<cv::Vec3b> scene_color_channels_;  ///< converted color values where each point corresponds to a row entry
  typename pcl::octree::OctreePointCloudSearch<PointTWithNormal>::Ptr octree_scene_downsampled_;
  boost::function<void(const boost::dynamic_bitset<> &, double, size_t)> visualize_cues_during_logger_;

  Eigen::VectorXi scene_pt_smooth_label_id_;  ///< stores a label for each point of the (downsampled) scene. Points
                                              /// belonging to the same smooth clusters, have the same label
  size_t max_smooth_label_id_ = 0;
  size_t num_evaluations_ = 0;  ///< counts how many times it evaluates a solution until finished verification

  // ----- MULTI-VIEW VARIABLES------
  std::vector<typename pcl::PointCloud<PointTWithNormal>::ConstPtr>
      occlusion_clouds_;  ///< scene clouds from multiple views (stored as organized point clouds)
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> absolute_camera_poses_;
  std::vector<boost::dynamic_bitset<>> model_is_present_in_view_;  ///< for each model this variable stores information
                                                                   /// in which view it is present (used to check for
  /// visible model points - default all true = static
  /// scene)

  boost::function<float(const Eigen::VectorXf &, const Eigen::VectorXf &)> color_dist_f_;

  template <typename ICP>
  void refinePose(HVRecognitionModel<PointT> &rm) const;

  cv::Mat img_boundary_distance_;  ///< saves for each pixel how far it is away from the boundary (taking into account
                                   /// extrinsics of the camera)

  boost::optional<Eigen::Matrix4f> transform_to_world_;  ///< optional affine transformation used to reject hypotheses
                                                         /// below floor

  OutlineVerification outline_verification_;
  DepthOutlinesParameter depth_outlines_param_;

  /**
   * @brief computeVisiblePoints Occlusion reasoning based on self-occlusion and occlusion from scene cloud(s). It first
   * renders the model cloud in the given pose onto the image plane and checks via z-buffering for each model point if
   * it is visible or self-occluded. The visible model cloud is then compared to the scene cloud for occlusion  caused
   * by the input scene.
   * @param rm recongition model
   */
  void computeVisibleModelPoints(HVRecognitionModel<PointT> &rm)
      const;  ///< computes the visible points of the model in the given pose and the provided depth map(s) of the scene

  /**
   * @brief computeVisibleOctreeNodes the visible point computation so far misses out points that are not visible just
   * because of the discretization from the image back-projection
   * for each pixel only one point falling on that pixel can be said to be visible with the approach so far.
   * Particularly in high-resolution models it is however possible that multiple points project on the same pixel and
   * they should be also visible.
   * if there is at least one visible point computed from the approach above. If so, the leaf node is counted visible
   * and all its containing points
   * are also set as visible.
   * @param rm recognition model
   */
  void computeVisibleOctreeNodes(HVRecognitionModel<PointT> &rm) const;

  void downsampleSceneCloud();  ///< downsamples the scene cloud

  void computePairwiseIntersection();  ///< computes the overlap of two visible points when projected to camera view

  void computeSmoothRegionOverlap();  ///< computes if two hypotheses explain the same smooth region

  void computeModelFitness(HVRecognitionModel<PointT> &rm) const;

  bool checkIfModelIsUnderFloor(HVRecognitionModel<PointT> &rm, const Eigen::Matrix4f &transform_to_world) const;

  void initialize();

  double evaluateSolution(const boost::dynamic_bitset<> &solution, bool &violates_smooth_region_check);

  void optimize();

  /**
   * @brief check if provided input is okay
   */
  void checkInput() const;

  /**
   * @brief this iterates through all poses and rejects hypotheses that are very similar.
   * Similar means that they describe the same object identity, their centroids align and the rotation is (almost) the
   * same.
   */
  void removeRedundantPoses();

  /**
   * @brief isOutlier remove hypotheses with a lot of outliers. Returns true if hypothesis is rejected.
   * @param rm
   * @return
   */
  bool isOutlier(const HVRecognitionModel<PointT> &rm) const;

  void cleanUp() {
    octree_scene_downsampled_.reset();
    occlusion_clouds_.clear();
    absolute_camera_poses_.clear();
    scene_sampled_indices_.reset();
    model_is_present_in_view_.clear();
    scene_cloud_downsampled_.reset();
    scene_w_normals_.reset();
    intersection_cost_.resize(0, 0);
    obj_hypotheses_groups_.clear();
    scene_pt_smooth_label_id_.resize(0);
    scene_color_channels_.clear();
    scene_pts_explained_solution_.clear();
    global_hypotheses_.clear();
    outline_verification_.resetScene();
  }

  /**
   * @brief extractEuclideanClustersSmooth
   */
  void extractEuclideanClustersSmooth();

  /**
   * @brief getFitness
   * @param c
   * @return
   */
  float getFitness(const ModelSceneCorrespondence &c) const {
    if (param_.ignore_color_even_if_exists_)
      return modelSceneNormalsCostTerm(c);
    else
      return modelSceneNormalsCostTerm(c) * scoreColorNormalized(c);
  }

  inline float scoreColor(float dist_color) const {
    return dist_color < param_.inlier_threshold_color_;
  }

  /**
   * @brief modelSceneColorCostTerm
   * @param model scene correspondence
   * @return
   */
  float scoreColorNormalized(const ModelSceneCorrespondence &c) const {
    return scoreColor(c.color_distance_);
  }

  inline float scoreNormals(float dotp) const {
    return dotp > param_.inlier_threshold_normals_dotp_;
  }

  /**
   * @brief modelSceneNormalsCostTerm
   * @param model scene correspondence
   * @return angle between corresponding surface normals (fliped such that they are pointing into the same direction)
   */
  float modelSceneNormalsCostTerm(const ModelSceneCorrespondence &c) const {
    return scoreNormals(c.normals_dotp_);
  }

  //    void
  //    computeLOffset( HVRecognitionModel<PointT> &rm ) const;

  float customColorDistance(const Eigen::VectorXf &color_a, const Eigen::VectorXf &color_b);

  /**
   * @brief customRegionGrowing constraint function which decides if two points are to be merged as one "smooth" cluster
   * @param seed_pt
   * @param candidate_pt
   * @param squared_distance
   * @return
   */
  bool customRegionGrowing(const PointTWithNormal &seed_pt, const PointTWithNormal &candidate_pt,
                           float squared_distance) const;

  static inline bool isFinite(const PointTWithNormal &p) {
    return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) && std::isfinite(p.normal_x) &&
           std::isfinite(p.normal_y) && std::isfinite(p.normal_z);
  }

  /**
   * @brief function to check whether or not some of the algorithms require the relationship between indices of the
   * downsampled scene cloud to the full (organized) input cloud
   * @return true if relationship is required
   */
  bool needFullToDownsampledSceneIndexRelation() const {
    return param_.check_smooth_clusters_;
  }

  float search_radius_;

 public:
  HypothesisVerification(const Intrinsics::ConstPtr &cam, const HV_Parameter &p = HV_Parameter())
  : param_(p), cam_(cam) {
    max_smooth_label_id_ = 0;

    switch (param_.color_comparison_method_) {
      case ColorComparisonMethod::CIE76:
        color_dist_f_ = computeCIE76;
        break;

      case ColorComparisonMethod::CIE94:
        color_dist_f_ = computeCIE94_DEFAULT;
        break;

      case ColorComparisonMethod::CIEDE2000:
        color_dist_f_ = computeCIEDE2000;
        break;

      case ColorComparisonMethod::CUSTOM:
        color_dist_f_ = boost::bind(&HypothesisVerification::customColorDistance, this, _1, _2);
        break;

      default:
        throw std::runtime_error("Color comparison method not defined!");
    }

    eps_angle_threshold_rad_ = pcl::deg2rad(param_.eps_angle_threshold_deg_);

    outline_verification_.setEnabled(p.outline_verification_);
    outline_verification_.setThreshold(p.outline_verification_threshold_);

    depth_outlines_param_ = p.depth_outlines_param_;
  }

  /**
   *  \brief Sets the scene cloud
   *  \param scene_cloud Point cloud representing the scene
   */
  void setSceneCloud(const typename pcl::PointCloud<PointTWithNormal>::ConstPtr &scene_cloud);

  /**
   * @brief set Occlusion Clouds And Absolute Camera Poses (used for multi-view recognition)
   * @param occlusion clouds
   * @param absolute camera poses
   */
  void setOcclusionCloudsAndAbsoluteCameraPoses(
      const std::vector<typename pcl::PointCloud<PointTWithNormal>::ConstPtr> &occ_clouds,
      const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &absolute_camera_poses) {
    for (size_t i = 0; i < occ_clouds.size(); i++)
      CHECK(occ_clouds[i]->isOrganized()) << "Occlusion clouds need to be organized!";

    occlusion_clouds_ = occ_clouds;
    absolute_camera_poses_ = absolute_camera_poses;
  }

  /**
   * @brief for each model this variable stores information in which view it is present
   * @param presence in model and view
   */
  void setVisibleCloudsForModels(const std::vector<boost::dynamic_bitset<>> &model_is_present_in_view) {
    model_is_present_in_view_ = model_is_present_in_view;
  }

  /**
   * @brief setHypotheses
   * @param ohs
   */
  void setHypotheses(std::vector<ObjectHypothesesGroup> &ohs);

  /**
   * @brief setModelDatabase
   * @param m_db model database
   */
  void setModelDatabase(const typename Source<PointT>::ConstPtr &m_db);

  /**
   * @brief set RGB Depth overlap mask
   * @param rgb_depth_overlap this mask has the same size as the camera image and tells us which pixels can have valid
   * depth pixels and which ones are not seen due to the physical displacement between RGB and depth sensor. Valid
   * pixels are set to 255, pixels that are outside depth camera's field of view are set to 0
   */
  void setRGBDepthOverlap(cv::InputArray &rgb_depth_overlap);

  /**
   * @brief visualizeModelCues visualizes the model cues during computation. Useful for debugging
   * @param vis_params visualization parameters
   */
  void visualizeModelCues(
      const PCLVisualizationParams::ConstPtr &vis_params = std::make_shared<PCLVisualizationParams>()) {
    vis_model_.reset(new HV_ModelVisualizer<PointT>(vis_params));
  }

  /**
   * @brief visualizeCues visualizes the cues during the computation and shows cost and number of evaluations. Useful
   * for debugging
   * @param vis_params visualization parameters
   */
  void visualizeCues(const PCLVisualizationParams::ConstPtr &vis_params = std::make_shared<PCLVisualizationParams>()) {
    vis_cues_.reset(new HV_CuesVisualizer<PointT>(vis_params));
  }

  /**
   * @brief visualizePairwiseCues visualizes the pairwise intersection of two hypotheses during computation. Useful for
   * debugging
   * @param vis_params visualization parameters
   */
  void visualizePairwiseCues(
      const PCLVisualizationParams::ConstPtr &vis_params = std::make_shared<PCLVisualizationParams>()) {
    vis_pairwise_.reset(new HV_PairwiseVisualizer<PointT>(vis_params));
  }

  /**
   *  \brief Function that performs the hypotheses verification
   *  This function modifies the values of mask_ and needs to be called after both scene and model have been added
   */
  void verify();

  /**
   * @brief getElapsedTimes
   * @return compuation time measurements for various components
   */
  std::vector<std::pair<std::string, float>> getElapsedTimes() const {
    return elapsed_time_;
  }

  /**
   * @brief setCamera set the camera used for z-buffering
   * @param cam camera parameters
   */
  void setCameraIntrinsics(const Intrinsics::ConstPtr &cam) {
    cam_ = cam;
  }

  /**
   * @brief setTransformToWorld set the transformation that transforms scene so floor is set with Z = 0
   * @param transform_to_world affine transformation
   */
  void setTransformToWorld(const boost::optional<Eigen::Matrix4f> &transform_to_world) {
    transform_to_world_ = transform_to_world;
  }

  /**
   * Can be used to enable/disable outline verification
   * @param is_enabled if true then it's enabled
   */
  void setOutlineVerification(bool is_enabled) {
    outline_verification_.setEnabled(is_enabled);
  }
};

template <typename PointT>
std::vector<std::pair<std::string, float>> HypothesisVerification<PointT>::elapsed_time_;

}  // namespace v4r
