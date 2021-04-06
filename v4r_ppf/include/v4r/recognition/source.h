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
 * @file source.h
 * @author Aitor Aldoma (aldoma@acin.tuwien.ac.at), Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
 * @date 2017
 * @brief
 *
 */

#pragma once

#include <v4r/recognition/model.h>

namespace v4r {

struct SourceParameter {
  std::string view_prefix_ = "cloud_";
  std::string pose_prefix_ = "pose_";
  std::string indices_prefix_ = "object_indices_";
  std::string view_folder_name_ =
      "views";  //< name of the folder containing the training views (point clouds) of the object
  std::string name_3D_model_ = "3D_model.pcd";  //< filename of the 3D object model
  std::string metadata_name_ = "metadata.yaml";
  bool has_categories_ =
      false;  //< if true, reads a model database used for classification, i.e. there is another top-level
  // folders for each category and inside each category folder there is the same structure as for instance recognition
};

/**
 * \brief Abstract data source class, manages filesystem, incremental training, etc.
 * \author Aitor Aldoma, Thomas Faeulhammer
 */
template <typename PointT>
class Source {
 private:
  SourceParameter param_;

  std::vector<typename Model<PointT>::Ptr> models_;  ///< all models

 public:
  Source(const SourceParameter &p = SourceParameter()) : param_(p) {}

  /**
   * @brief Source
   * @param model_database_path path to object model database. This class assumes that each object is stored in a
   * separate folder. Each of these folders has a folder "/views" with training views in it.
   * Each training view has a pointcloud which filename begins with the string in variable view_prefix,
   * object indices that indicate the object which filename begins with the string in variable indices_prefix,
   * a 4x4 homogenous camera pose that aligns the training views into a common coordinate system when multiplied with
   * each other which filename begins with the string in variable pose_prefix
   * @param object_instances_to_load vector of object models to load from model_database_path. If empty, all objects
   * in directory will be loaded.
   */
  void init(const boost::filesystem::path &model_database_path,
            const std::vector<std::string> &object_instances_to_load = {});

  /**
   * \brief Get the generated model
   * \return returns all generated models
   */
  std::vector<typename Model<PointT>::Ptr> getModels() const {
    return models_;
  }

  /**
   * @brief getModelById
   * @param class_id unique identifier of the model category
   * @param model_id unique identifier of the model instance
   * @return model pointer
   */
  typename Model<PointT>::ConstPtr getModelById(const std::string &class_id, const std::string &instance_id) const;

  /**
   * @brief addModel add a model to the database
   * @param m model
   */
  void addModel(const typename Model<PointT>::Ptr m) {
    models_.push_back(m);
  }

  /**
   * @brief cleans up data stored in each models' training view
   * @param keep_computed_properties whether or not input filenames and computed properties such as centroid,
   * eigen_pose_alignment, etc. are being kept
   */
  void cleanUpTrainingData(bool keep_computed_properties = false) {
    for (auto &m : models_)
      m->cleanUpTrainingData(keep_computed_properties);
  }

  typedef std::shared_ptr<Source<PointT>> Ptr;
  typedef std::shared_ptr<Source<PointT> const> ConstPtr;
};
}  // namespace v4r
