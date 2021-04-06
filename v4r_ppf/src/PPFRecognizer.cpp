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
 * @file PPFRecognizer.cpp
 * @author Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
 * @date 2017
 * @brief
 *
 */

#include <iostream>
#include <sstream>

#include <glog/logging.h>
#include <omp.h>
#include <pcl/common/time.h>
#include <pcl/filters/passthrough.h>
#include <pcl/registration/icp.h>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <PPFRecognizer.h>
// #include <v4r/keypoints/all_headers.h>
// #include <v4r/ml/all_headers.h>
#include <ppf_recognition_pipeline.h>
#include <v4r/registration/noise_model_based_cloud_integration.h>
#include <v4r/segmentation/plane_utils.h>

namespace po = boost::program_options;

namespace v4r {

namespace apps {

template <typename PointT>
void PPFRecognizer<PointT>::validate() {
  CHECK(bf::exists(models_dir_)) << "Given model directory (" << models_dir_ << ") does not exist!";

  const auto model_folders = v4r::io::getFoldersInDirectory(models_dir_);
  CHECK(!model_folders.empty()) << "Given model directory (" << models_dir_ << " does not contain models!";

  CHECK(std::all_of(param_.object_models_.cbegin(), param_.object_models_.cend(),
                    [&model_folders, this](const std::string &model) {
                      if (std::find(model_folders.cbegin(), model_folders.cend(), model) == model_folders.cend()) {
                        LOG(ERROR) << "Given object model to load \"" << model
                                   << "\" is not present in the model directory (" << this->models_dir_ << ")!";
                        return false;
                      }
                      return true;
                    }));
}

template <typename PointT>
void PPFRecognizer<PointT>::setup(bool force_retrain, bool use_color) {
  validate();

  try {
    *param_.cam_ = Intrinsics::load(param_.camera_calibration_file_.string());
  } catch (const std::runtime_error &e) {
    LOG(WARNING) << "Failed to load camera calibration file from " << param_.camera_calibration_file_.string()
                 << "! Will use Primesense default camera intrinsics parameters!" << std::endl;
    *param_.cam_ = Intrinsics::PrimeSense();
  }

  // ==== FILL OBJECT MODEL DATABASE ==== ( assumes each object is in a separate folder named after the object and
  // contains and "views" folder with the training views of the object)
  SourceParameter source_param;
  model_database_.reset(new Source<PointT>(source_param));
  model_database_->init(models_dir_, param_.object_models_);

  // ====== SETUP PPF RECOGNITION PIPELINE ======
  typename PPFRecognitionPipeline<PointT>::Ptr ppf_rec_pipeline(
          new PPFRecognitionPipeline<PointT>(param_.ppf_rec_pipeline_));

  ppf_rec_pipeline->setModelDatabase(model_database_);
  // multipipeline->setModelDatabase(model_database_);

  /* if (param_.use_multiview_) {
    typename MultiviewRecognizer<PointT>::Ptr mv_rec(new v4r::MultiviewRecognizer<PointT>(param_.multiview_));
    mv_rec->setSingleViewRecognitionPipeline(ppf_rec_pipeline);
    mv_rec->setModelDatabase(model_database_);
    mrec_ = mv_rec;
  } else {
    mrec_ = ppf_rec_pipeline;
  } */
  mrec_ = ppf_rec_pipeline;

  mrec_->initialize(models_dir_, force_retrain, param_.object_models_);
  model_database_->cleanUpTrainingData(true);

  if (!param_.skip_verification_) {
    hv_.reset(new HypothesisVerification<PointT>(param_.cam_, param_.hv_));

    cv::Mat_<uchar> img_mask = cv::imread(param_.rgb_depth_overlap_image_.string(), CV_LOAD_IMAGE_GRAYSCALE);
    if (img_mask.data)
      hv_->setRGBDepthOverlap(img_mask);
    else
      LOG(WARNING) << "No camera depth registration mask provided. Assuming all pixels have valid depth.";

    hv_->setModelDatabase(model_database_);
  }
}

template <typename PointT>
std::vector<ObjectHypothesesGroup> PPFRecognizer<PointT>::recognize(const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
    const std::vector<std::string> &obj_models_to_search_tmp, const boost::optional<pcl::PointCloud<pcl::Normal>::Ptr> &cloud_normals,
    const boost::optional<Eigen::Matrix4f> &transform_to_world, cv::InputArray &region_of_interest) {

  omp_set_nested(1);
  std::vector<ObjectHypothesesGroup> generated_object_hypotheses;
  typename pcl::PointCloud<PointT>::Ptr processed_cloud(new pcl::PointCloud<PointT>(*cloud));

  // check if object models to search for exist in model database
  for (const auto &id : obj_models_to_search_tmp) {
    if (!model_database_->getModelById("", id))
      LOG(ERROR) << "Requested object model \"" << id << "\" is not found in model database " << models_dir_;
  }

  std::vector<std::string> obj_models_to_search;
  for (const auto &m : model_database_->getModels()) {
    if (obj_models_to_search_tmp.empty() || std::find(obj_models_to_search_tmp.begin(), obj_models_to_search_tmp.end(),
                                                      m->id_) != obj_models_to_search_tmp.end()) {
      obj_models_to_search.push_back(m->id_);
    }
  }

  if (obj_models_to_search.empty()) {
    LOG(ERROR) << "No valid objects to search for specified!";
    return {};
  } else {
    std::stringstream info_txt;
    info_txt << "Searching for following object model(s): ";
    for (const auto &m : obj_models_to_search)
      info_txt << m << ", ";
    LOG(INFO) << info_txt.str();
  }

  omp_set_nested(1);
#pragma omp parallel
  {
#pragma omp single
    {
      pcl::StopWatch t_total;
      elapsed_time_.clear();

      if (param_.cam_->w != cloud->width || param_.cam_->h != cloud->height) {
        LOG(WARNING) << "Input cloud has different resolution (" << cloud->width << "x" << cloud->height
                     << ") than resolution stated in camera calibration file (" << param_.cam_->w << "x"
                     << param_.cam_->h << "). Will adjust camera calibration file accordingly.";
        param_.cam_->adjustToSize(cloud->width, cloud->height);
        LOG(INFO) << "Adapted intrinsics " << *param_.cam_;
      }

      pcl::PointCloud<pcl::Normal>::Ptr normals;
      if (mrec_->needNormals() || hv_) {
          if (cloud_normals.is_initialized()) {
              //normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>(cloud_normals.get()));
              normals = cloud_normals.get();
          } else {
              //compute normals
              pcl::StopWatch t;
              const std::string time_desc("Computing normals");
              normals = computeNormals<PointT>(processed_cloud, param_.normal_estimator_param_);
              double time = t.getTime();
              VLOG(1) << time_desc << " took " << time << " ms.";
              elapsed_time_.push_back(std::pair<std::string, float>(time_desc, time));
          }
          mrec_->setSceneNormals(normals);
      }
      else {  // since we only work with PointTWithNormal types for some components, we need to have some dummy
          // normals at least
          normals.reset(new pcl::PointCloud<pcl::Normal>);
      }

      bool do_multiview = param_.use_multiview_ && param_.use_multiview_hv_;
      if (do_multiview && !transform_to_world) {
        LOG(ERROR) << "Multiview recognition enabled but no camera pose provided!";
        do_multiview = false;
      }

      if (!param_.skip_verification_ && !do_multiview) {
#pragma omp task
        {
          typename pcl::PointCloud<PointTWithNormal>::Ptr cloud_w_normals(new pcl::PointCloud<PointTWithNormal>);
          pcl::concatenateFields(*cloud, *normals, *cloud_w_normals);
          hv_->setSceneCloud(cloud_w_normals);
        }
      }

      Eigen::Vector4f support_plane;  //< plane which is believed to support the object (either floor plane or some
                                      // higher plane parallel to it)
      if (transform_to_world) {
        for (PointT &p : processed_cloud->points) {
          float z_world = transform_to_world.get().row(2).dot(p.getVector4fMap());

          // filter points based on height above ground
          if (z_world < param_.min_height_above_ground_ || z_world > param_.max_height_above_ground_) {
            p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
          }
        }

        // set support plane to floor plane (assumed to correspond to z=0 in world reference frame)
        const Eigen::Vector4f search_axis_world = Eigen::Vector4f::UnitZ();
        support_plane = transform_to_world.get().transpose() * search_axis_world;  // transformed to camera ref. frame
      }

      if (region_of_interest.empty()) {
        if (param_.remove_planes_) {
          pcl::StopWatch t;
          const std::string time_desc("Removing planes");

          // check for largest object model diameter and only remove planes larger than this maximum diameter
          float largest_model_diameter = 0.f;
          for (const auto &m_id : obj_models_to_search) {
            const auto m = model_database_->getModelById("", m_id);
            const auto model_diameter = m->getDiameter();
            if (model_diameter > largest_model_diameter)
              largest_model_diameter = model_diameter;
          }
          LOG(INFO) << "Largest model diameter is " << largest_model_diameter << "m. ";

          param_.plane_filter_.min_plane_diameter_ =
              largest_model_diameter * param_.max_model_diameter_to_min_plane_ratio_;

          VLOG(1) << "setting plane filter min diameter = " << param_.plane_filter_.min_plane_diameter_;

          v4r::apps::CloudSegmenter<PointT> plane_extractor(
              param_.plane_filter_);  ///< cloud segmenter for plane removal (if enabled)
          plane_extractor.initialize();
          plane_extractor.setNormals(normals);
          plane_extractor.segment(processed_cloud, transform_to_world);
          processed_cloud = plane_extractor.getProcessedCloud();
          support_plane = plane_extractor.getSelectedPlane();

          double time = t.getTime();
          VLOG(1) << time_desc << " took " << time << " ms.";
          elapsed_time_.push_back(std::pair<std::string, float>(time_desc, time));
        }
      } else {
        // remove points outside ROI
        const cv::Mat roi = region_of_interest.getMat();
        for (int v = 0; v < roi.rows; v++) {
          for (int u = 0; u < roi.cols; u++) {
            if (roi.at<unsigned char>(v, u) == 0) {
              PointT &p = processed_cloud->at(u, v);
              p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
            }
          }
        }

        if (!param_.skip_verification_) {
          LOG(INFO) << "ROI set, disabled outline verification";
          hv_->setOutlineVerification(false);
        }
      }

      mrec_->setTablePlane(support_plane);

      // filter points based on distance to camera
      for (PointT &p : processed_cloud->points) {
        if (pcl::isFinite(p) && p.getVector3fMap().norm() > param_.chop_z_)
          p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
      }

      {
        pcl::StopWatch t;
        const std::string time_desc("Generation of object hypotheses");

        mrec_->setInputCloud(processed_cloud);
        if (transform_to_world)
          mrec_->setTransformToWorld(transform_to_world.get());
        mrec_->recognize(obj_models_to_search);
        generated_object_hypotheses = mrec_->getObjectHypothesis();

        double time = t.getTime();
        VLOG(1) << time_desc << " took " << time << " ms.";
        elapsed_time_.push_back(std::pair<std::string, float>(time_desc, time));
        std::vector<std::pair<std::string, float>> elapsed_times_rec = mrec_->getElapsedTimes();
        elapsed_time_.insert(elapsed_time_.end(), elapsed_times_rec.begin(), elapsed_times_rec.end());
      }

      if (param_.skip_verification_ && param_.icp_iterations_) {
        for (size_t ohg_id = 0; ohg_id < generated_object_hypotheses.size(); ohg_id++) {
          for (size_t oh_id = 0; oh_id < generated_object_hypotheses[ohg_id].ohs_.size(); oh_id++) {
            ObjectHypothesis::Ptr &oh = generated_object_hypotheses[ohg_id].ohs_[oh_id];

            const auto m = model_database_->getModelById("", oh->model_id_);
            DownsamplerParameter ds_param;
            ds_param.resolution_ = 0.005f;  // TODO: make this a parameter
            const auto model_cloud = m->getAssembled(ds_param);

            const Eigen::Matrix4f hyp_tf_2_global = oh->pose_refinement_ * oh->transform_;
            typename pcl::PointCloud<PointTWithNormal>::Ptr model_cloud_aligned(new pcl::PointCloud<PointTWithNormal>);
            pcl::copyPointCloud(*model_cloud, *model_cloud_aligned);  // TODO make ICP use PointTWithNormal
            pcl::transformPointCloud(*model_cloud_aligned, *model_cloud_aligned, hyp_tf_2_global);

            //point-to-plane ICP = ICPwithNormals
            //remove nans and downsample scene and model cloud
            typename pcl::PointCloud<PointTWithNormal>::Ptr scene_w_normals(new pcl::PointCloud<PointTWithNormal>);
            pcl::concatenateFields(*processed_cloud, *normals, *scene_w_normals);
            std::vector<int> nan_ind;
            pcl::removeNaNFromPointCloud(*scene_w_normals, *scene_w_normals, nan_ind);
            Downsampler ds(ds_param);
            typename pcl::PointCloud<PointTWithNormal>::Ptr scene_cloud_downsampled_(new pcl::PointCloud<PointTWithNormal>);
            scene_cloud_downsampled_ = ds.downsample<PointTWithNormal>(scene_w_normals);
            pcl::IterativeClosestPointWithNormals<PointTWithNormal, PointTWithNormal> icp;
            typename pcl::search::KdTree<PointTWithNormal>::Ptr kdtree_scene(new pcl::search::KdTree<PointTWithNormal>);
            kdtree_scene->setInputCloud(scene_w_normals);
            icp.setInputSource(model_cloud_aligned);
            icp.setInputTarget(scene_w_normals);
            icp.setTransformationEpsilon(param_.icp_transf_eps_);
            icp.setMaximumIterations(static_cast<int>(param_.icp_iterations_));
            icp.setMaxCorrespondenceDistance(param_.icp_max_corr_dist_);
            icp.setSearchMethodTarget(kdtree_scene, true);
            pcl::PointCloud<PointTWithNormal> aligned_visible_model;
            icp.align(aligned_visible_model);

//            typename pcl::search::KdTree<PointT>::Ptr kdtree_scene(new pcl::search::KdTree<PointT>);
//            kdtree_scene->setInputCloud(processed_cloud);
//            pcl::IterativeClosestPoint<PointT, PointT> icp;
//            icp.setInputSource(model_cloud_aligned);
//            icp.setInputTarget(processed_cloud);
//            icp.setTransformationEpsilon(param_.icp_transf_eps_);
//            icp.setMaximumIterations(static_cast<int>(param_.icp_iterations_));
//            icp.setMaxCorrespondenceDistance(param_.icp_max_corr_dist_);
//            icp.setSearchMethodTarget(kdtree_scene, true);
//            pcl::PointCloud<PointT> aligned_visible_model;
//            icp.align(aligned_visible_model);

            Eigen::Matrix4f pose_refinement;
            if (icp.hasConverged()) {
              pose_refinement = icp.getFinalTransformation();
              oh->pose_refinement_ = pose_refinement * oh->pose_refinement_;
            } else
              LOG(WARNING) << "ICP did not converge" << std::endl;
          }
        }
      }

      // Hypothesis verification
      if (!param_.skip_verification_) {
#pragma omp taskwait
        hv_->setTransformToWorld(transform_to_world);
        hv_->setHypotheses(generated_object_hypotheses);

        if (do_multiview) {
          NMBasedCloudIntegrationParameter nm_int_param;
          nm_int_param.min_points_per_voxel_ = 1;
          nm_int_param.octree_resolution_ = 0.002f;

          View v;
          v.cloud_ = cloud;
          v.processed_cloud_ = processed_cloud;
          v.camera_pose_ = transform_to_world.get();
          v.cloud_normals_ = normals;

          size_t num_views = std::min<size_t>(param_.multiview_max_views_, views_.size() + 1);
          LOG(INFO) << "Running multi-view recognition over " << num_views;

          views_.push_back(v);

          std::vector<typename pcl::PointCloud<PointTWithNormal>::ConstPtr> views(
              num_views);  ///< all views in multi-view sequence
          std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> camera_poses(
              num_views);  ///< all absolute camera poses in multi-view sequence

          {
            pcl::StopWatch t;
            const std::string time_desc("Noise model based cloud integration");
            NMBasedCloudIntegration<PointT> nmIntegration(nm_int_param, param_.cam_);
            size_t tmp_id = 0;
            for (size_t v_id = views_.size() - num_views; v_id < views_.size(); v_id++) {
              const View &vv = views_[v_id];
              nmIntegration.addView(vv.processed_cloud_, vv.cloud_normals_, vv.camera_pose_);
              typename pcl::PointCloud<PointTWithNormal>::Ptr view_w_normals(new pcl::PointCloud<PointTWithNormal>);
              pcl::concatenateFields(*vv.cloud_, *vv.cloud_normals_, *view_w_normals);
              views[tmp_id] = view_w_normals;
              camera_poses[tmp_id] = vv.camera_pose_;
              tmp_id++;
            }
            nmIntegration.compute(registered_scene_cloud_);  // is in global reference frame
            normals = nmIntegration.getOutputNormals();

            double time = t.getTime();
            VLOG(1) << time_desc << " took " << time << " ms.";
            elapsed_time_.push_back(std::pair<std::string, float>(time_desc, time));
          }


          const Eigen::Matrix4f tf_global2cam = transform_to_world.get().inverse();

          typename pcl::PointCloud<PointT>::Ptr registerd_scene_cloud_latest_camera_frame(new pcl::PointCloud<PointT>);
          pcl::transformPointCloud(*registered_scene_cloud_, *registerd_scene_cloud_latest_camera_frame, tf_global2cam);
          pcl::PointCloud<pcl::Normal>::Ptr normals_aligned(new pcl::PointCloud<pcl::Normal>);
          v4r::transformNormals(*normals, *normals_aligned, tf_global2cam);

          typename pcl::PointCloud<PointTWithNormal>::Ptr cloud_w_normals(new pcl::PointCloud<PointTWithNormal>);
          pcl::concatenateFields(*registerd_scene_cloud_latest_camera_frame, *normals_aligned, *cloud_w_normals);
          hv_->setSceneCloud(cloud_w_normals);

          // describe the clouds with respect to the most current view
          std::transform(camera_poses.begin(), camera_poses.end(), camera_poses.begin(),
                         [&tf_global2cam](auto &p) -> Eigen::Matrix4f { return tf_global2cam * p; });

          hv_->setOcclusionCloudsAndAbsoluteCameraPoses(views, camera_poses);
        }

        pcl::StopWatch t;
        const std::string time_desc("Verification of object hypotheses");
        hv_->verify();
        double time = t.getTime();
        VLOG(1) << time_desc << " took " << time << " ms.";
        elapsed_time_.push_back(std::pair<std::string, float>(time_desc, time));

        std::vector<std::pair<std::string, float>> hv_elapsed_times = hv_->getElapsedTimes();
        elapsed_time_.insert(elapsed_time_.end(), hv_elapsed_times.begin(), hv_elapsed_times.end());
      }

      if (param_.remove_planes_ && param_.remove_non_upright_objects_) {
        for (size_t ohg_id = 0; ohg_id < generated_object_hypotheses.size(); ohg_id++) {
          for (size_t oh_id = 0; oh_id < generated_object_hypotheses[ohg_id].ohs_.size(); oh_id++) {
            ObjectHypothesis::Ptr &oh = generated_object_hypotheses[ohg_id].ohs_[oh_id];

            if (!oh->is_verified_)
              continue;

            const Eigen::Matrix4f tf = oh->pose_refinement_ * oh->transform_;
            const Eigen::Vector3f translation = tf.block<3, 1>(0, 3);
            double dist2supportPlane = fabs(v4r::dist2plane(translation, support_plane));
            const Eigen::Vector3f z_orientation = tf.block<3, 3>(0, 0) * Eigen::Vector3f::UnitZ();
            float dotp =
                z_orientation.dot(support_plane.head(3)) / (support_plane.head(3).norm() * z_orientation.norm());
            VLOG(1) << "dotp for model " << oh->model_id_ << ": " << dotp;

            if (dotp < 0.8f) {
              oh->is_verified_ = false;
              VLOG(1) << "Rejected " << oh->model_id_ << " because it is not standing upgright (dot-product = " << dotp
                      << ")!";
            }
            if (dist2supportPlane > 0.03) {
              oh->is_verified_ = false;
              VLOG(1) << "Rejected " << oh->model_id_
                      << " because object origin is too far away from support plane = " << dist2supportPlane << ")!";
            }
          }
        }
      }

      double time_total = t_total.getTime();

      std::stringstream info;
      size_t num_detected = 0;
      for (size_t ohg_id = 0; ohg_id < generated_object_hypotheses.size(); ohg_id++) {
        for (const auto &oh : generated_object_hypotheses[ohg_id].ohs_) {
          if (oh->is_verified_) {
            num_detected++;
            const std::string &model_id = oh->model_id_;
            const Eigen::Matrix4f &tf = oh->transform_;
            float confidence = oh->confidence_;
            info << "" << model_id << " (confidence: " << std::fixed << std::setprecision(2) << confidence
                 << ") with pose:" << std::endl
                 << std::setprecision(5) << tf << std::endl
                 << std::endl;
          }
        }
      }

      std::stringstream rec_info;
      rec_info << "Detected " << num_detected << " object(s) in " << time_total << "ms" << std::endl << info.str();
      LOG(INFO) << rec_info.str();

      if (!FLAGS_logtostderr)
        std::cout << rec_info.str();
    }
  }
  return generated_object_hypotheses;
}

template <typename PointT>
void PPFRecognizer<PointT>::resetMultiView() {
  /* if (param_.use_multiview_) {
    views_.clear();

    typename v4r::MultiviewRecognizer<PointT>::Ptr mv_rec =
        std::dynamic_pointer_cast<v4r::MultiviewRecognizer<PointT>>(mrec_);
    if (mrec_)
      mv_rec->clear();
    else
      LOG(ERROR) << "Cannot reset multi-view recognizer because given recognizer is not a multi-view recognizer!";
  } */
}

template class PPFRecognizer<pcl::PointXYZRGB>;
}  // namespace apps
}  // namespace v4r
