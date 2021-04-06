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
 * @file model.h
 * @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
 * @date 2016
 * @brief
 *
 */

#pragma once

#include <boost/filesystem.hpp>
#include <boost/serialization/vector.hpp>

#include <pcl/common/centroid.h>
#include <pcl/point_cloud.h>

#include <v4r/common/Clustering.h>
#include <v4r/common/downsampler.h>
#include <v4r/common/point_types.h>
#include <v4r/geometry/normals.h>

namespace bf = boost::filesystem;

namespace v4r {

/**
 * @brief class to describe a training view of the object model
 * @author Thomas Faeulhammer
 * @date Oct 2016
 */
template <typename PointT>
struct TrainingView {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typename pcl::PointCloud<PointT>::ConstPtr cloud_;    ///< point cloud of view
  pcl::PointCloud<pcl::Normal>::ConstPtr normals_;      ///< normals for point cloud of view
  Eigen::Matrix4f pose_ = Eigen::Matrix4f::Identity();  ///< corresponding camera pose (s.t. multiplying the individual
                                                        ///< clouds with these transforms
                                                        /// bring it into a common coordinate system)
  bf::path filename_ = "";                              ///< cloud filename of the training view
  bf::path pose_filename_ = "";                         ///< pose filename of the training view
  bf::path indices_filename_ = "";                      ///< object mask/indices filename of the training view
  std::vector<int> indices_;                            ///< corresponding object indices
  Eigen::Vector3f elongation_ = Eigen::Vector3f::Zero();  ///< elongations in meter for each dimension
  Eigen::Matrix4f eigen_pose_alignment_ =
      Eigen::Matrix4f::Identity();  ///< rigid transform that aligns principal axis of object with x,y and z axis
  Eigen::Vector4f centroid_ = Eigen::Vector4f(0, 0, 0, 1);  ///< centroid of object in view
  Eigen::Vector3f centroid_diff_to_complete_model_centroid_ =
      Eigen::Vector3f::Zero();  ///< distance of the object's view centroid to the
                                ///< centroid of the full 3d object model

  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    (void)version;
    ar &cloud_;
    ar &pose_;
    ar &filename_;
    ar &pose_filename_;
    ar &indices_filename_;
    ar &indices_;
    ar &centroid_;
  }

  void clearClouds() {
    cloud_.reset();
    normals_.reset();
  }

  using Ptr = std::shared_ptr<TrainingView<PointT>>;
  using ConstPtr = std::shared_ptr<TrainingView<PointT> const>;
};

enum DetectionCues { VISUAL, GEOMETRY };

struct ModelProperties {
  std::vector<bool> symmetry_xyz_ = {false, false, false};  ///< indicates if the object is symmetric around the
                                                            ///< plane defined by surface normals x, y, z respectively
                                                            ///< example: x = true -> symmetrical around yz plane
  std::vector<bool> rotational_invariance_xyz_ = {false, false, false};  ///< indicates if the object is invariant to
                                                                         ///< rotations around x, y or z axis (of the
                                                                         ///< object coordinate system) respectively
  std::vector<DetectionCues> detection_cues_ = {
      DetectionCues::VISUAL, DetectionCues::GEOMETRY};  ///< indicates which cues can be used to detect the object

  /**
   * construct object model properties from YAML file
   * @param metadata_file yaml file which specifies the properties
   */
  ModelProperties(const bf::path &metadata_file = "");
};

/**
 * @brief Class representing a recognition model
 */
template <typename PointT>
class Model {
 private:
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    (void)version;
    ar &class_;
    ar &id_;
    ar &*cluster_props_;
  }

  using PointTWithNormal = v4r::add_normal_t<PointT>;

  /**
   * @brief Point cloud representation of object model. Note that this actually stores redundant information to
   * avoid additional computational cost in extracting cloud and normals separately from the cloud.
   */

  mutable typename std::map<int, typename pcl::PointCloud<PointTWithNormal>::ConstPtr>
      voxelized_assembled_;  ///< cached point clouds of object models downsampled to a specific resolution in
                             ///< millimeter (for speed-up purposes)

  typename pcl::PointCloud<PointTWithNormal>::ConstPtr all_assembled_;  ///< full resolution object model
  typename pcl::PointCloud<PointTWithNormal>::Ptr convex_hull_points_;  ///< convex hull of object model

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  std::vector<typename TrainingView<PointT>::Ptr> views_;
  std::string class_, id_;
  Eigen::Vector4f minPoint_;  ///< defines the 3D bounding box of the object model
  Eigen::Vector4f maxPoint_;  ///< defines the 3D bounding box of the object model
  ModelProperties properties_;
  Cluster::Ptr cluster_props_;  ///< elongation of the object model along the three principal
                                ///< axes. First element corresponds to largest elongation,
                                ///< third element to smallest.
  float diameter_ = 0.f;        ///< model diameter in meter (maximum distance between two points of the object)

  Model() {}

  bool operator==(const Model &other) const {
    return (id_ == other.id_) && (class_ == other.class_);
  }

  /**
   * @brief addTrainingView
   * @param tv training view
   */
  void addTrainingView(const typename TrainingView<PointT>::Ptr tv) {
    views_.push_back(tv);
  }

  std::vector<typename TrainingView<PointT>::Ptr> getTrainingViews() const {
    return views_;
  }

  /**
   * @brief cleans up data stored in each training view
   * @param keep_computed_properties whether or not input filenames and computed properties such as centroid,
   * eigen_pose_alignment, etc. are being kept
   */
  void cleanUpTrainingData(bool keep_computed_properties = false) {
    if (keep_computed_properties) {
      for (auto &v : views_)
        v->clearClouds();
    } else {
      views_.clear();
    }
  }

  /**
   * @brief initialize initializes the model creating 3D models and so on
   * @param model_filename path to the 3D model (if path does not exist or is empty, 3D model will be created by simple
   * accumulation of points in the training views)
   * @param normal_estimator normal estimator parameters to estimate normals in the individual training views (if
   * normals do not exist yet and 3D model needs to be created)
   */
  void initialize(const bf::path &model_filename = "",
                  const NormalEstimatorParameter &ne_param = NormalEstimatorParameter());

  /**
   * @brief return model point cloud (with normals)
   * @return point cloud model of the object
   */
  typename pcl::PointCloud<PointTWithNormal>::ConstPtr getAssembled() const {
    return all_assembled_;
  }

  typename pcl::PointCloud<PointTWithNormal>::ConstPtr getConvexHull() const {
    return convex_hull_points_;
  }

  /**
   * @brief return model point cloud (with normals) in desired resolution
   * @param ds_param Downsampling parameter
   * @param keep_downsampled_cloud if true, caches the downsampled point cloud and returns this cloud from memory next
   * time the assembly function is called with the same resolution
   * @return point cloud model of the object
   */
  typename pcl::PointCloud<PointTWithNormal>::ConstPtr getAssembled(const DownsamplerParameter &ds_param,
                                                                    bool keep_downsampled_cloud = true) const {
    int resolution_mm = static_cast<int>(ds_param.resolution_ * 1000.f);
    assert(resolution_mm >= 0);

    const auto it = voxelized_assembled_.find(resolution_mm);
    if (it != voxelized_assembled_.end()) {
      return it->second;
    } else {
      Downsampler ds(ds_param);
      const auto downsampled_cloud = ds.downsample<PointTWithNormal>(all_assembled_);
      if (keep_downsampled_cloud)
        voxelized_assembled_[resolution_mm] = downsampled_cloud;
      return downsampled_cloud;
    }
  }

  /**
   * @return model diameter in meter (maximum distance between two points of the object)
   */
  float getDiameter() const {
    return diameter_;
  }

  typedef std::shared_ptr<Model<PointT>> Ptr;
  typedef std::shared_ptr<Model<PointT> const> ConstPtr;
};

}  // namespace v4r
