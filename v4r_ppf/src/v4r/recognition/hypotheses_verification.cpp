#include <glog/logging.h>
#include <v4r/common/color_transforms.h>
#include <v4r/common/histogram.h>
#include <v4r/common/metrics.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/occlusion_reasoning.h>
#include <v4r/common/pcl_utils.h>
#include <v4r/common/time.h>
#include <v4r/geometry/normals.h>
#include <v4r/recognition/hypotheses_verification.h>
#include <v4r/segmentation/segmenter_conditional_euclidean.h>
#include <v4r/common/impl/zbuffering.hpp>

#include <pcl/common/time.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/pcl_config.h>
#include <pcl/registration/gicp.h>
#include <pcl_1_8/keypoints/uniform_sampling.h>

#include <omp.h>
#include <opencv2/opencv.hpp>

namespace {
Eigen::Vector3f lab_conversion(const cv::Vec3b &lab_cv) {
  Eigen::Vector3f lab;
  lab << lab_cv[0] / 2.55f, lab_cv[1] - 128.f, lab_cv[2] - 128.f;
  return lab;
}

template <typename PointT>
bool checkNormals(const pcl::PointCloud<PointT> &cloud_with_normals) {
  bool is_okay = true;
  for (const auto &p : cloud_with_normals) {
    if (fabs(p.getNormalVector3fMap().norm() - 1.f) > 0.05f) {
      VLOG(3) << "Normal is not normalized (norm: " << p.getNormalVector3fMap().norm() << ")";
      is_okay = false;
    }
    if (p.getNormalVector4fMap()[3] + 0.f != 0.f) {
      VLOG(3) << "Last element of 4-dimensional normal vector is not 0! (" << p.getNormalVector4fMap()[3] << ")";
      is_okay = false;
    }
    if (!std::isfinite(p.normal_x) || !std::isfinite(p.normal_y) || !std::isfinite(p.normal_z)) {
      VLOG(3) << "Normal is not finite!";
      is_okay = false;
    }
  }
  return is_okay;
}
}  // namespace

namespace v4r {

template <typename PointT>
bool HypothesisVerification<PointT>::customRegionGrowing(const PointTWithNormal &seed_pt,
                                                         const PointTWithNormal &candidate_pt,
                                                         float squared_distance) const {
  (void)squared_distance;

  // float curvature_threshold = param_.curvature_threshold_;
  float eps_angle_threshold_rad = eps_angle_threshold_rad_;

  if (!std::isfinite(seed_pt.getNormalVector4fMap()(0)) || std::isnan(seed_pt.getNormalVector4fMap()(0)) ||
      !std::isfinite(candidate_pt.getNormalVector4fMap()(0)) || std::isnan(candidate_pt.getNormalVector4fMap()(0)))
    return false;

  if (param_.z_adaptive_) {
    float mult = std::max(seed_pt.z, 1.f);
    //            mult *= mult;
    // curvature_threshold = param_.curvature_threshold_ * mult;
    eps_angle_threshold_rad = eps_angle_threshold_rad_ * mult;
  }

  if (seed_pt.curvature > param_.curvature_threshold_)
    return false;

  if (candidate_pt.curvature > param_.curvature_threshold_)
    return false;

  float intensity_a = .2126f * seed_pt.r + .7152f * seed_pt.g + .0722f * seed_pt.b;
  float intensity_b = .2126f * candidate_pt.r + .7152f * candidate_pt.g + .0722f * candidate_pt.b;

  if (fabs(intensity_a - intensity_b) > 5.f)
    return false;

  float dotp = seed_pt.getNormalVector4fMap().dot(candidate_pt.getNormalVector4fMap());
  if (dotp < cos(eps_angle_threshold_rad))
    return false;

  return true;
}

template <typename PointT>
float HypothesisVerification<PointT>::customColorDistance(const Eigen::VectorXf &color_a,
                                                          const Eigen::VectorXf &color_b) {
  float L_dist = (color_a(0) - color_b(0)) * (color_a(0) - color_b(0));
  CHECK(L_dist >= 0.f && L_dist <= 1.f);
  L_dist /= 20;
  float AB_dist = (color_a.tail(2) - color_b.tail(2)).norm();  // ( param_.color_sigma_ab_ * param_.color_sigma_ab_ );
  CHECK(AB_dist >= 0.f && AB_dist <= 1.f);
  return L_dist + AB_dist;
}

template <typename PointT>
void HypothesisVerification<PointT>::computeVisibleModelPoints(HVRecognitionModel<PointT> &rm) const {
  // EASY_BLOCK("compute model occlusion by scene");
  const auto m = m_db_->getModelById("", rm.oh_->model_id_);
  CHECK(m) << "Model of type \""
           << ""
           << "\" and id \"" << rm.oh_->model_id_ << "\" not found!";
  const auto model_cloud = m->getAssembled();  // use full resolution for rendering

  const Eigen::Matrix4f hyp_tf_2_global = rm.oh_->pose_refinement_ * rm.oh_->transform_;
  typename pcl::PointCloud<PointTWithNormal>::Ptr model_cloud_aligned(new pcl::PointCloud<PointTWithNormal>);
  pcl::transformPointCloudWithNormals(*model_cloud, *model_cloud_aligned, hyp_tf_2_global);

  boost::dynamic_bitset<> image_mask_mv(model_cloud->size(), 0);
  rm.image_mask_.resize(occlusion_clouds_.size(), boost::dynamic_bitset<>(occlusion_clouds_[0]->size(), 0));

  constexpr bool visualize_visibility_computation = false;

  for (size_t view = 0; view < occlusion_clouds_.size(); view++) {
    // project into respective view
    typename pcl::PointCloud<PointTWithNormal>::Ptr aligned_model_cloud(new pcl::PointCloud<PointTWithNormal>);
    const Eigen::Matrix4f tf = absolute_camera_poses_[view].inverse();
    pcl::transformPointCloudWithNormals(*model_cloud_aligned, *aligned_model_cloud, tf);

    // check self-occlusion
    ZBufferingParameter zBparam;
    zBparam.do_noise_filtering_ = false;
    zBparam.do_smoothing_ = false;
    zBparam.use_normals_ = true;
    ZBuffering<PointTWithNormal> zbuf(cam_, zBparam);
    auto organized_model_cloud_to_be_filtered = zbuf.renderPointCloud(*aligned_model_cloud, 1);
    //        std::vector<int> kept_indices = zbuf.getKeptIndices();
    const Eigen::MatrixXi index_map = zbuf.getIndexMap();

    if (visualize_visibility_computation) {
      static pcl::visualization::PCLVisualizer vis("self-occlusion");
      static int vp0, vp1, vp2;
      vis.removeAllPointClouds();
      vis.createViewPort(0, 0, 0.33, 1, vp0);
      vis.createViewPort(0.33, 0, 0.66, 1, vp1);
      vis.createViewPort(0.66, 0, 1, 1, vp2);
      vis.addPointCloud<PointTWithNormal>(aligned_model_cloud, "input", vp0);
      vis.addText("aligned object model", 10, 10, 1., 1., 1., "vp1_text", vp0);
      vis.addCoordinateSystem(0.2, "co0", vp0);
      vis.addCoordinateSystem(0.2, "co1", vp1);
      vis.addCoordinateSystem(0.2, "co2", vp2);
      pcl::visualization::PointCloudColorHandlerCustom<PointTWithNormal> gray(scene_cloud_downsampled_, 128, 128, 128);
      vis.addPointCloud<PointTWithNormal>(scene_cloud_downsampled_, gray, "input_rm_vp_model_", vp0);
      vis.addPointCloud<PointTWithNormal>(organized_model_cloud_to_be_filtered, "organized", vp1);
      vis.addText("organized model cloud", 10, 10, 1., 1., 1., "organized_cloud_to_be_filtered", vp1);
      boost::dynamic_bitset<> image_mask_sv(model_cloud->points.size(), 0);
      for (size_t u = 0; u < organized_model_cloud_to_be_filtered->width; u++) {
        for (size_t v = 0; v < organized_model_cloud_to_be_filtered->height; v++) {
          int original_idx = index_map(v, u);

          if (original_idx < 0)
            continue;
          image_mask_sv.set(original_idx);
        }
      }

      typename pcl::PointCloud<PointTWithNormal>::Ptr visible_object_cloud(new pcl::PointCloud<PointTWithNormal>);
      pcl::copyPointCloud(*aligned_model_cloud, image_mask_sv, *visible_object_cloud);
      vis.addPointCloud<PointTWithNormal>(visible_object_cloud, "vis_cloud2", vp2);
      std::cout << "Number visible object points (after self-occlusion check): " << visible_object_cloud->size()
                << std::endl;
      vis.addText("visible cloud (check with indices)", 10, 10, 1., 1., 1., "visible_cloud", vp2);
      vis.spin();
    }

    // now check occlusion by other objects
    const auto &pt_is_visible = rm.image_mask_[view] =
        computeVisiblePoints(*organized_model_cloud_to_be_filtered, *occlusion_clouds_[view], param_.occlusion_thres_);

    for (size_t v = 0; v < organized_model_cloud_to_be_filtered->height; v++) {
      for (size_t u = 0; u < organized_model_cloud_to_be_filtered->width; u++) {
        int idx = v * organized_model_cloud_to_be_filtered->width + u;

        if (!img_boundary_distance_.empty() &&
            img_boundary_distance_.at<float>(v, u) < param_.min_px_distance_to_image_boundary_)
          continue;

        if (pt_is_visible[idx]) {
          int original_idx = index_map(v, u);

          if (original_idx < 0)
            continue;

          const Eigen::Vector3f viewray = aligned_model_cloud->points[original_idx].getVector3fMap().normalized();
          const Eigen::Vector3f normal = aligned_model_cloud->points[original_idx].getNormalVector3fMap();

          float dotp = viewray.dot(normal);

          if (dotp > -param_.min_dotproduct_model_normal_to_viewray_)
            continue;

          image_mask_mv.set(original_idx);
        }
      }
    }

    if (visualize_visibility_computation) {
      static pcl::visualization::PCLVisualizer vis("occlusion reasoning with scene");
      static int vp0, vp1, vp2;
      vis.removeAllPointClouds();
      vis.createViewPort(0, 0, 0.33, 1, vp0);
      vis.createViewPort(0.33, 0, 0.66, 1, vp1);
      vis.createViewPort(0.66, 0, 1, 1, vp2);
      vis.addPointCloud<PointTWithNormal>(organized_model_cloud_to_be_filtered, "model_cloud", vp0);
      vis.addText("rendered object model", 10, 10, 1., 1., 1., "organized_model_cloud_to_be_filtered", vp0);
      vis.addPointCloud<PointTWithNormal>(occlusion_clouds_[view], "occluder_cloud", vp1);
      vis.addText("scene occlusion cloud", 10, 10, 1., 1., 1., "scene_occlusion_cloud", vp1);
      typename pcl::PointCloud<PointTWithNormal>::Ptr visible_cloud(new pcl::PointCloud<PointTWithNormal>);
      pcl::copyPointCloud(*organized_model_cloud_to_be_filtered, pt_is_visible, *visible_cloud);
      std::cout << "Number visible object points (after scene occlusion check): " << visible_cloud->size() << std::endl;
      vis.addPointCloud<PointTWithNormal>(visible_cloud, "vis_cloud", vp2);
      vis.addText("visible object points (after scene occlusion check)", 10, 10, 1., 1., 1., "visible_cloud", vp2);
      vis.spin();
    }
  }

  std::vector<int> visible_indices_tmp_full = createIndicesFromMask<int>(image_mask_mv);
  typename pcl::PointCloud<PointTWithNormal>::Ptr visible_cloud_full(new pcl::PointCloud<PointTWithNormal>);
  pcl::copyPointCloud(*model_cloud, visible_indices_tmp_full, *visible_cloud_full);

  // downsample
  pcl_1_8::UniformSampling<PointTWithNormal> us;
  us.setRadiusSearch(param_.scene_downsampler_param_.resolution_);
  us.setInputCloud(visible_cloud_full);
  pcl::PointCloud<int> sampled_indices;
  us.compute(sampled_indices);

  rm.visible_indices_.resize(sampled_indices.size());
  for (size_t i = 0; i < sampled_indices.size(); i++) {
    int idx = visible_indices_tmp_full[sampled_indices[i]];
    rm.visible_indices_[i] = idx;
  }

  rm.visible_cloud_.reset(new pcl::PointCloud<PointTWithNormal>);
  pcl::copyPointCloud(*model_cloud_aligned, rm.visible_indices_, *rm.visible_cloud_);

  if (visualize_visibility_computation) {
    static pcl::visualization::PCLVisualizer vis("final visible object model");
    static int vp0, vp1, vp2, vp3;
    vis.addCoordinateSystem(0.2, "co0", vp0);
    vis.addCoordinateSystem(0.2, "co1", vp1);
    vis.addCoordinateSystem(0.2, "co2", vp2);
    vis.addCoordinateSystem(0.2, "co3", vp3);
    vis.createViewPort(0, 0, 0.25, 1, vp0);
    vis.createViewPort(0.25, 0, 0.5, 1, vp1);
    vis.createViewPort(0.5, 0, 0.75, 1., vp2);
    vis.createViewPort(0.75, 0, 1., 1., vp3);
    vis.removeAllPointClouds();
    vis.addPointCloud<PointTWithNormal>(scene_cloud_downsampled_, "scene", vp0);
    vis.addText("scene", 10, 10, 1., 1., 1., "scene", vp0);
    vis.addPointCloud<PointTWithNormal>(model_cloud_aligned, "aligned_model", vp1);
    vis.addText("aligned model", 10, 10, 1., 1., 1., "model_cloud_aligned", vp1);
    vis.addPointCloud<PointTWithNormal>(rm.visible_cloud_, "visible_model", vp2);
    vis.addText("visible cloud", 10, 10, 1., 1., 1., "visible_cloud", vp2);

    typename pcl::PointCloud<PointTWithNormal>::Ptr vis_cloud2(new pcl::PointCloud<PointTWithNormal>);
    pcl::copyPointCloud(*model_cloud_aligned, rm.visible_indices_by_octree_, *vis_cloud2);
    vis.addPointCloud<PointTWithNormal>(vis_cloud2, "visible_model_by_octree", vp3);
    vis.addText("visible cloud by octree", 10, 10, 1., 1., 1., "visible_cloud_octree", vp3);

    std::cout << "Number visible object points (after all occlusion check): " << rm.visible_cloud_->size() << std::endl;
    std::cout << "Number visible object points (after all occlusion check and using octree): " << vis_cloud2->size()
              << std::endl;
    vis.spin();
  }
}

template <typename PointT>
void HypothesisVerification<PointT>::computeVisibleOctreeNodes(HVRecognitionModel<PointT> &rm) const {
  ScopeTime t("compute visible octree nodes");
  // EASY_BLOCK("compute visible octree nodes");
  const boost::dynamic_bitset<> visible_mask = v4r::createMaskFromIndices(rm.visible_indices_, rm.num_pts_full_model_);
  const auto octree_it = octree_model_representation_.find(rm.oh_->model_id_);
  CHECK(octree_it != octree_model_representation_.end());

  boost::dynamic_bitset<> visible_leaf_mask(rm.num_pts_full_model_, 0);
#if PCL_VERSION_COMPARE(>=, 1, 9, 0)
  for (auto leaf_it = octree_it->second->leaf_depth_begin(); leaf_it != octree_it->second->leaf_depth_end();
       ++leaf_it) {
#else
  for (auto leaf_it = octree_it->second->leaf_begin(); leaf_it != octree_it->second->leaf_end(); ++leaf_it) {
#endif
    pcl::octree::OctreeContainerPointIndices &container = leaf_it.getLeafContainer();
    std::vector<int> indexVector;
    container.getPointIndices(indexVector);
    if (std::any_of(indexVector.begin(), indexVector.end(),
                    [&visible_mask](const int idx) { return visible_mask.test(idx); })) {
      // leaf node is visible
      for (int idx : indexVector)
        visible_leaf_mask.set(idx);
    }
  }
  rm.visible_indices_by_octree_ = createIndicesFromMask<int>(visible_leaf_mask);
}

template <typename PointT>
template <typename ICP>
void HypothesisVerification<PointT>::refinePose(HVRecognitionModel<PointT> &rm) const {
  // EASY_BLOCK("Pose refinement");

  const auto m = m_db_->getModelById("", rm.oh_->model_id_);
  const Eigen::Vector4f max_model_point = m->maxPoint_;
  const Eigen::Vector4f min_model_point = m->minPoint_;

  typename pcl::PointCloud<PointTWithNormal>::Ptr cropped_scene(new pcl::PointCloud<PointTWithNormal>);
  {
    // EASY_BLOCK("Crop scene around model bbox");

    pcl::CropBox<PointTWithNormal> crop_filter;
    crop_filter.setInputCloud(scene_cloud_downsampled_);

    // to compensate rotation estimation error, bounding box will be a bit bigger.
    Eigen::Vector4f bbox_diagonal_dir = max_model_point - min_model_point;
    auto diagonal_length = bbox_diagonal_dir.norm();
    bbox_diagonal_dir /= diagonal_length;
    constexpr float scale_weight = 0.2f;
    constexpr float margin_scale = scale_weight * 0.5f * (2.f - 1.41421356237f);
    const float margin = margin_scale * diagonal_length;
    crop_filter.setMin(min_model_point - margin * bbox_diagonal_dir);
    crop_filter.setMax(max_model_point + margin * bbox_diagonal_dir);

    // before cropping, each scene point is transformed to the model space
    const Eigen::Matrix4f model2scene = rm.oh_->transform_;
    Eigen::Affine3f scene2model;
    scene2model.matrix() = model2scene.inverse();
    crop_filter.setTransform(scene2model);
    crop_filter.filter(*cropped_scene);
  }

  if (cropped_scene->size() < 20 || rm.visible_cloud_->size() < 25) {
    VLOG(2) << "cannot refine pose of model " << rm.oh_->model_id_ << " because there are not enough points left! "
            << "cropped scene size: " << cropped_scene->size() << " and visible cloud: " << rm.visible_cloud_->size();
    return;
  }

  typename pcl::search::KdTree<PointTWithNormal>::Ptr cropped_scene_kdtree(new pcl::search::KdTree<PointTWithNormal>);
  cropped_scene_kdtree->setInputCloud(cropped_scene);

  if (VLOG_IS_ON(3)) {
    if (!checkNormals(*cropped_scene))
      LOG(WARNING) << "Something is wrong with cropped scene!";
    if (!checkNormals(*rm.visible_cloud_))
      LOG(WARNING) << "Something is wrong with visible model cloud!";
  }

  ICP icp;
  icp.setInputTarget(cropped_scene);
  icp.setSearchMethodTarget(cropped_scene_kdtree, true);
  icp.setTransformationEpsilon(1e-5);
  icp.setEuclideanFitnessEpsilon(5e-3);
  icp.setMaximumIterations(param_.icp_iterations_);
  icp.setMaxCorrespondenceDistance(param_.icp_max_correspondence_);
  icp.setInputSource(rm.visible_cloud_);
  pcl::PointCloud<PointTWithNormal> aligned_visible_model;
  icp.align(aligned_visible_model);

  if (icp.hasConverged()) {
    Eigen::Matrix4f model2scene_pose_refinement = icp.getFinalTransformation();
    rm.oh_->pose_refinement_ = model2scene_pose_refinement * rm.oh_->pose_refinement_;
  } else
    LOG(WARNING) << "ICP did not converge" << std::endl;

  constexpr bool kEnableVisualisation = false;  // won't work if OpenMP is enabled

  // cppcheck-suppress knownConditionTrueFalse
  if (kEnableVisualisation) {
    static pcl::visualization::PCLVisualizer vis_tmp;
    static int vp1, vp2, vp3;
    vis_tmp.removeAllPointClouds();
    vis_tmp.removeAllShapes();
    vis_tmp.createViewPort(0, 0, 0.33, 1, vp1);
    vis_tmp.createViewPort(0.33, 0, 0.66, 1, vp2);
    vis_tmp.createViewPort(0.66, 0, 1, 1, vp3);

    typename pcl::PointCloud<PointT>::Ptr scene_cloud_vis(new pcl::PointCloud<PointT>),
        model_cloud(new pcl::PointCloud<PointT>), whole_scene(new pcl::PointCloud<PointT>);
    pcl::copyPointCloud(*cropped_scene, *scene_cloud_vis);
    pcl::copyPointCloud(*scene_cloud_downsampled_, *whole_scene);
    scene_cloud_vis->sensor_orientation_ = Eigen::Quaternionf::Identity();
    scene_cloud_vis->sensor_origin_ = Eigen::Vector4f::Zero(4);

    pcl::copyPointCloud(*rm.visible_cloud_, *model_cloud);

    vis_tmp.addPointCloud(model_cloud, "model1", vp1);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> gray(scene_cloud_vis, 255, 128, 128);
    vis_tmp.addPointCloud(scene_cloud_vis, gray, "scene1", vp1);
    vis_tmp.addText("before ICP", 10, 10, 20, 1, 1, 1, "before_ICP", vp1);

    vis_tmp.addPointCloud(scene_cloud_vis, gray, "scene2", vp2);
    typename pcl::PointCloud<PointT>::Ptr model_refined(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*model_cloud, *model_refined, rm.oh_->pose_refinement_);
    vis_tmp.addText("after ICP", 10, 10, 20, 1, 1, 1, "after_ICP", vp2);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> green(model_refined, 128, 255, 128);
    vis_tmp.addPointCloud(model_refined, green, "model_refined", vp2);

    Eigen::Matrix4f model2scene = rm.oh_->transform_;
    pcl::PointXYZ p0, p1;
    p0.getVector4fMap() = model2scene * min_model_point;
    p1.getVector4fMap() = model2scene * max_model_point;
    vis_tmp.addSphere(p0, 0.015, 0, 255, 0, "min", vp3);
    vis_tmp.addSphere(p1, 0.015, 0, 255, 0, "max", vp3);
    vis_tmp.addPointCloud(model_refined, "model2", vp3);
    vis_tmp.addPointCloud(whole_scene, "whole_scene", vp3);
    vis_tmp.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "whole_scene", vp3);
    vis_tmp.spin();
  }
}

template <typename PointT>
void HypothesisVerification<PointT>::downsampleSceneCloud() {
  // EASY_BLOCK("Downsampling of scene cloud");
  ScopeTime t("Downsampling scene cloud");
  if (!scene_cloud_downsampled_)
    scene_cloud_downsampled_.reset(new pcl::PointCloud<PointTWithNormal>());

  if (param_.scene_downsampler_param_.resolution_ <= 0) {
    scene_cloud_downsampled_ = scene_w_normals_;
  } else {
    if (needFullToDownsampledSceneIndexRelation()) {
      if (param_.scene_downsampler_param_.method_ != DownsamplerParameter::Method::UNIFORM) {
        LOG(WARNING) << "Downsampling method " << param_.scene_downsampler_param_.method_
                     << "!=" << DownsamplerParameter::Method::UNIFORM << ". Will use UNIFORM downsampling!";
        param_.scene_downsampler_param_.method_ = DownsamplerParameter::Method::UNIFORM;
      }
      Downsampler ds(param_.scene_downsampler_param_);
      scene_cloud_downsampled_ = ds.downsample<PointTWithNormal>(scene_w_normals_);
      scene_sampled_indices_ = ds.getExtractedIndices();

      scene_indices_map_ = Eigen::VectorXi::Constant(scene_w_normals_->size(), -1);
      for (size_t i = 0; i < scene_sampled_indices_->size(); i++) {
        scene_indices_map_[scene_sampled_indices_->operator[](i)] = i;
      }
    } else {
      Downsampler ds(param_.scene_downsampler_param_);
      scene_cloud_downsampled_ = ds.downsample<PointTWithNormal>(scene_w_normals_);
    }

    VLOG(1) << "Downsampled scene cloud from " << scene_w_normals_->size() << " to " << scene_cloud_downsampled_->size()
            << " points using " << param_.scene_downsampler_param_.method_ << " with a resolution of "
            << param_.scene_downsampler_param_.resolution_ << "m.";
  }
}

template <typename PointT>
void HypothesisVerification<PointT>::search() {
  // EASY_BLOCK("Searching for optimal solution");
  pcl::StopWatch t;

  // set initial solution to hypotheses that do not have any intersection and not lie on same smooth cluster as other
  // hypothesis
  boost::dynamic_bitset<> initial_solution(global_hypotheses_.size(), 0);
  for (size_t i = 0; i < global_hypotheses_.size(); i++) {
    const auto rm = global_hypotheses_[i];
    if (intersection_cost_.row(i).sum() == 0 &&
        (!param_.check_smooth_clusters_ || smooth_region_overlap_.row(i).sum() == 0) &&
        (!param_.check_smooth_clusters_ || !rm->violates_smooth_cluster_check_)) {
      initial_solution.set(i);
    }
  }

  bool initial_solution_violates_smooth_region_check;
  double initial_cost = evaluateSolution(initial_solution, initial_solution_violates_smooth_region_check);
  evaluated_solutions_.insert(initial_solution.to_ulong());
  if (!param_.check_smooth_clusters_ || !initial_solution_violates_smooth_region_check) {
    best_solution_.solution_ = initial_solution;
    best_solution_.cost_ = initial_cost;
  } else {
    best_solution_.solution_ = boost::dynamic_bitset<>(global_hypotheses_.size(), 0);
    best_solution_.cost_ = std::numeric_limits<double>::max();
  }

  // now do a local search by enabling one hyphotheses at a time and also multiple hypotheses if they are on the same
  // smooth cluster
  bool everything_checked = false;
  evaluated_solutions_.insert(initial_solution.to_ulong());
  while (!everything_checked) {
    everything_checked = true;
    std::vector<size_t> solutions_to_be_tested;
    // flip one bit at a time
    for (size_t i = 0; i < best_solution_.solution_.size(); i++) {
      boost::dynamic_bitset<> current_solution = best_solution_.solution_;
      if (current_solution[i])
        continue;

      current_solution.flip(i);

      size_t solution_uint = current_solution.to_ulong();
      solutions_to_be_tested.push_back(solution_uint);

      // also test solutions with two new hypotheses which both describe the same smooth cluster. This should avoid
      // rejection of them if the objects are e.g. stacked together and only one smooth cluster for the stack is found.
      /// TODO: also implement checks for more than two hypotheses describing the same smooth cluster!
      if (param_.check_smooth_clusters_ && smooth_region_overlap_.row(i).sum() > 0) {
        for (size_t j = 0; j < best_solution_.solution_.size(); j++) {
          boost::dynamic_bitset<> ss = current_solution;
          if (smooth_region_overlap_(i, j) > 0 && !current_solution[j]) {
            ss.set(j);
            size_t s_uint = ss.to_ulong();
            solutions_to_be_tested.push_back(s_uint);
          }
        }
      }
    }

    for (size_t s_uint : solutions_to_be_tested) {
      // check if already evaluated
      if (evaluated_solutions_.find(s_uint) != evaluated_solutions_.end())
        continue;

      boost::dynamic_bitset<> s(global_hypotheses_.size(), s_uint);
      bool violates_smooth_region_check;
      double cost = evaluateSolution(s, violates_smooth_region_check);
      evaluated_solutions_.insert(s.to_ulong());
      if (cost < best_solution_.cost_ && (!param_.check_smooth_clusters_ || !violates_smooth_region_check)) {
        best_solution_.cost_ = cost;
        best_solution_.solution_ = s;
        everything_checked = false;
      }
    }
  }
  VLOG(1) << "Local search with " << num_evaluations_ << " evaluations took " << t.getTime() << " ms" << std::endl;
}

template <typename PointT>
double HypothesisVerification<PointT>::evaluateSolution(const boost::dynamic_bitset<> &solution,
                                                        bool &violates_smooth_region_check) {
  // EASY_BLOCK("evaluate solution");
  scene_pts_explained_solution_.clear();
  scene_pts_explained_solution_.resize(scene_cloud_downsampled_->points.size());

  for (size_t i = 0; i < global_hypotheses_.size(); i++) {
    if (!solution[i])
      continue;

    const typename HVRecognitionModel<PointT>::Ptr rm = global_hypotheses_[i];
    for (Eigen::SparseVector<float>::InnerIterator it(rm->scene_explained_weight_); it; ++it)
      scene_pts_explained_solution_[it.row()].push_back(PtFitness(it.value(), i));
  }

  for (auto spt_it = scene_pts_explained_solution_.begin(); spt_it != scene_pts_explained_solution_.end(); ++spt_it)
    std::sort(spt_it->begin(), spt_it->end());

  double scene_fit = 0., duplicity = 0.;
  Eigen::Array<bool, Eigen::Dynamic, 1> scene_pt_is_explained(scene_cloud_downsampled_->size());
  scene_pt_is_explained.setConstant(scene_cloud_downsampled_->size(), false);

  for (size_t s_id = 0; s_id < scene_cloud_downsampled_->size(); s_id++) {
    const std::vector<PtFitness> &s_pt = scene_pts_explained_solution_[s_id];
    if (!s_pt.empty()) {
      scene_fit += s_pt.back().fit_;  // uses the maximum value for scene explanation
      scene_pt_is_explained(s_id) = true;
    }

    if (s_pt.size() > 1)                        // two or more hypotheses explain the same scene point
      duplicity += s_pt[s_pt.size() - 2].fit_;  // uses the second best explanation
  }

  violates_smooth_region_check = false;
  if (param_.check_smooth_clusters_) {
    int max_label = scene_pt_smooth_label_id_.maxCoeff();
    for (int i = 1; i < max_label; i++)  // label "0" is for points not belonging to any smooth region
    {
      Eigen::Array<bool, Eigen::Dynamic, 1> s_pt_in_region = (scene_pt_smooth_label_id_.array() == i);
      Eigen::Array<bool, Eigen::Dynamic, 1> explained_pt_in_region =
          (s_pt_in_region.array() && scene_pt_is_explained.array());
      size_t num_explained_pts_in_region = explained_pt_in_region.count();
      size_t num_pts_in_smooth_regions = s_pt_in_region.count();

      if (num_explained_pts_in_region > param_.min_pts_smooth_cluster_to_be_epxlained_ &&
          (float)(num_explained_pts_in_region) / num_pts_in_smooth_regions < param_.min_ratio_cluster_explained_) {
        violates_smooth_region_check = true;
      }
    }
  }

  double cost = -(log(scene_fit) - param_.clutter_regularizer_ * duplicity);

  // VLOG(2) << "Evaluation of solution " << solution
  //       << (violates_smooth_region_check ? " violates smooth region check!" : "") << " cost: " << cost;

  num_evaluations_++;

  if (vis_cues_) {
    vis_cues_->visualize(this, solution, cost);
  }

  return cost;  // return the dual to our max problem
}

template <typename PointT>
void HypothesisVerification<PointT>::computeSmoothRegionOverlap() {
  // EASY_BLOCK("compute smooth region overlap");
  smooth_region_overlap_ = Eigen::MatrixXi::Zero(global_hypotheses_.size(), global_hypotheses_.size());

  for (size_t i = 1; i < global_hypotheses_.size(); i++) {
    HVRecognitionModel<PointT> &rm_a = *global_hypotheses_[i];
    for (size_t j = 0; j < i; j++) {
      const HVRecognitionModel<PointT> &rm_b = *global_hypotheses_[j];

      if (rm_a.violates_smooth_cluster_check_ || rm_b.violates_smooth_cluster_check_) {
        size_t num_overlap = (rm_a.on_smooth_cluster_ & rm_b.on_smooth_cluster_).count();
        smooth_region_overlap_(i, j) = smooth_region_overlap_(j, i) = static_cast<int>(num_overlap);
      }
    }
  }
}

template <typename PointT>
void HypothesisVerification<PointT>::computePairwiseIntersection() {
  // EASY_BLOCK("compute pairwise intersection");
  intersection_cost_ = Eigen::MatrixXf::Zero(global_hypotheses_.size(), global_hypotheses_.size());

  for (size_t i = 1; i < global_hypotheses_.size(); i++) {
    const HVRecognitionModel<PointT> &rm_a = *global_hypotheses_[i];
    for (size_t j = 0; j < i; j++) {
      const HVRecognitionModel<PointT> &rm_b = *global_hypotheses_[j];

      size_t num_intersections = 0, total_rendered_points = 0;

      for (size_t view = 0; view < rm_a.image_mask_.size(); view++) {
        for (size_t px = 0; px < rm_a.image_mask_[view].size(); px++) {
          if (rm_a.image_mask_[view][px] && rm_b.image_mask_[view][px])
            num_intersections++;

          if (rm_a.image_mask_[view][px] || rm_b.image_mask_[view][px])
            total_rendered_points++;
        }
      }

      float conflict_cost = static_cast<float>(num_intersections) / total_rendered_points;
      intersection_cost_(i, j) = intersection_cost_(j, i) = conflict_cost;
    }
  }
}

template <typename PointT>
void HypothesisVerification<PointT>::setHypotheses(std::vector<ObjectHypothesesGroup> &ohs) {
  obj_hypotheses_groups_.clear();
  obj_hypotheses_groups_.resize(ohs.size());
  for (size_t i = 0; i < obj_hypotheses_groups_.size(); i++) {
    const ObjectHypothesesGroup &ohg = ohs[i];

    obj_hypotheses_groups_[i].resize(ohg.ohs_.size());
    for (size_t jj = 0; jj < ohg.ohs_.size(); jj++) {
      ObjectHypothesis::Ptr oh = ohg.ohs_[jj];
      obj_hypotheses_groups_[i][jj].reset(new HVRecognitionModel<PointT>(oh));

      const auto m = m_db_->getModelById("", oh->model_id_);
      const auto model_cloud = m->getAssembled();
      obj_hypotheses_groups_[i][jj]->num_pts_full_model_ = model_cloud->size();
    }
  }
}

template <typename PointT>
bool HypothesisVerification<PointT>::isOutlier(const HVRecognitionModel<PointT> &rm) const {
  float visible_ratio = rm.visible_indices_by_octree_.size() / (float)rm.num_pts_full_model_;
  float thresh = param_.min_fitness_ + (param_.min_fitness_ - param_.min_fitness_high_) * (visible_ratio - 0.5f) /
                                           (0.5f - param_.min_visible_ratio_);
  float min_fitness_threshold = std::max<float>(param_.min_fitness_, std::min<float>(param_.min_fitness_high_, thresh));

  bool is_rejected = rm.oh_->confidence_ < min_fitness_threshold;

  VLOG(1) << rm.oh_->class_id_ << " " << rm.oh_->model_id_ << (is_rejected ? " is rejected" : "")
          << " with visible ratio of : " << visible_ratio << " and fitness " << rm.oh_->confidence_ << " (by thresh "
          << min_fitness_threshold << ")";
  return is_rejected;
}

template <typename PointT>
void HypothesisVerification<PointT>::removeRedundantPoses() {
  ScopeTime t("Removing similar poses");
  // EASY_BLOCK("remove redundant poses");
  for (size_t i = 0; i < obj_hypotheses_groups_.size(); i++) {
    for (size_t j = 0; j < obj_hypotheses_groups_[i].size(); j++) {
      const HVRecognitionModel<PointT> &rm_a = *obj_hypotheses_groups_[i][j];

      if (!rm_a.isRejected()) {
        const Eigen::Matrix4f pose_a = rm_a.oh_->pose_refinement_ * rm_a.oh_->transform_;
        const Eigen::Vector4f centroid_a = pose_a.block<4, 1>(0, 3);
        const Eigen::Matrix3f rot_a = pose_a.block<3, 3>(0, 0);

        for (size_t ii = i; ii < obj_hypotheses_groups_.size(); ii++) {
          for (size_t jj = 0; jj < obj_hypotheses_groups_[ii].size(); jj++) {
            if (i == ii && jj <= j)
              continue;

            HVRecognitionModel<PointT> &rm_b = *obj_hypotheses_groups_[ii][jj];
            const Eigen::Matrix4f pose_b = rm_b.oh_->pose_refinement_ * rm_b.oh_->transform_;
            const Eigen::Vector4f centroid_b = pose_b.block<4, 1>(0, 3);
            const Eigen::Matrix3f rot_b = pose_b.block<3, 3>(0, 0);
            double dist = (centroid_a - centroid_b).norm();
            const double dist_r = pcl::rad2deg(getMinGeodesicDistance(rot_a, rot_b));

            if ((dist < param_.min_Euclidean_dist_between_centroids_) &&
                (dist_r < param_.min_angular_degree_dist_between_hypotheses_)) {
              rm_b.rejected_due_to_similar_hypothesis_exists_ = true;

              VLOG(1) << rm_b.oh_->class_id_ << " " << rm_b.oh_->model_id_
                      << " is rejected because a similar hypothesis already exists." << std::endl
                      << " centroids: " << centroid_a.transpose() << ", " << centroid_b.transpose() << "; ";
            }
          }
        }
      }
    }
  }
}

template <typename PointT>
void HypothesisVerification<PointT>::extractEuclideanClustersSmooth() {
  // EASY_BLOCK("extract smooth Euclidean clusters");
  boost::function<bool(const PointTWithNormal &, const PointTWithNormal &, float)> custom_f =
      boost::bind(&HypothesisVerification<PointT>::customRegionGrowing, this, _1, _2, _3);

  SegmenterParameter ces_param;
  ces_param.min_cluster_size_ = param_.min_points_;
  ces_param.max_cluster_size_ = std::numeric_limits<int>::max();
  ces_param.z_adaptive_ = param_.z_adaptive_;
  ces_param.distance_threshold_ = param_.cluster_tolerance_;

  ConditionalEuclideanSegmenter<PointTWithNormal> ces(ces_param);
  if (scene_w_normals_->isOrganized())
    ces.setInputCloud(scene_w_normals_);
  else
    ces.setInputCloud(scene_cloud_downsampled_);

  ces.setConditionFunction(custom_f);
  ces.segment();
  std::vector<std::vector<int>> clusters;
  ces.getSegmentIndices(clusters);

  // convert to downsample scene cloud indices
  if (scene_w_normals_->isOrganized()) {
    size_t kept_clusters = 0;
    for (size_t i = 0; i < clusters.size(); i++) {
      size_t kept = 0;
      for (size_t j = 0; j < clusters[i].size(); j++) {
        int sidx_original = clusters[i][j];
        int index_downsampled_scene = scene_indices_map_[sidx_original];
        if (index_downsampled_scene >= 0) {
          clusters[i][kept] = index_downsampled_scene;
          kept++;
        }
      }
      clusters[i].resize(kept);

      if (clusters[i].size() >= param_.min_points_) {
        clusters[kept_clusters] = clusters[i];
        kept_clusters++;
      }
    }
    clusters.resize(kept_clusters);
  }

  scene_pt_smooth_label_id_ = Eigen::VectorXi::Zero(scene_cloud_downsampled_->size());
  for (size_t i = 0; i < clusters.size(); i++) {
    for (int sidx : clusters[i]) {
      scene_pt_smooth_label_id_(sidx) = i + 1;  // label "0" for points not belonging to any smooth region
    }
  }
  max_smooth_label_id_ = clusters.size();
}

template <typename PointT>
bool HypothesisVerification<PointT>::checkIfModelIsUnderFloor(HVRecognitionModel<PointT> &rm,
                                                              const Eigen::Matrix4f &transform_to_world) const {
  const auto m = m_db_->getModelById("", rm.oh_->model_id_);
  auto convex_hull_points = m->getConvexHull();
  const Eigen::Matrix4f model_to_world = transform_to_world * rm.oh_->pose_refinement_ * rm.oh_->transform_;
  const float min_z = param_.floor_z_min_;
  rm.rejected_due_to_be_under_floor_ = std::any_of(convex_hull_points->begin(), convex_hull_points->end(),
                                                   [min_z, &model_to_world](const auto &pt) -> bool {
                                                     float z_level = (model_to_world * pt.getVector4fMap())[2];
                                                     return z_level < min_z;
                                                   });
  return rm.rejected_due_to_be_under_floor_;
}

// template<typename PointT, typename PointT>
// void
// HypothesisVerification<PointT>::computeLOffset(HVRecognitionModel<PointT> &rm)const
//{
//    Eigen::VectorXf color_s ( rm.scene_explained_weight_.nonZeros() );

//    size_t kept=0;
//    for (Eigen::SparseVector<float>::InnerIterator it(rm.scene_explained_weight_); it; ++it)
//    {
//        int sidx = it.index();
//        color_s(kept++) = scene_color_channels_(sidx,0);
//    }

//    Eigen::VectorXf color_new = specifyHistogram( rm.pt_color_.col( 0 ), color_s, 100, 0.f, 100.f );
//    rm.pt_color_.col( 0 ) = color_new;
//}

template <typename PointT>
void HypothesisVerification<PointT>::setRGBDepthOverlap(cv::InputArray &rgb_depth_overlap) {
  if (rgb_depth_overlap.empty()) {
    LOG(WARNING) << "Depth registration mask not set. Using the whole image!";
    img_boundary_distance_.release();
  } else {
    // EASY_BLOCK("Computing distance transform");
    ScopeTime t("Computing distance transform to image boundary.");
    cv::distanceTransform(rgb_depth_overlap, img_boundary_distance_, CV_DIST_L2, 5);
  }
}

template <typename PointT>
void HypothesisVerification<PointT>::initialize() {
  global_hypotheses_.clear();

  // check if provided rgb_depth_overlap mask is the same size as RGB image
  if (!img_boundary_distance_.empty()) {
    CHECK(static_cast<int>(cam_->w) == img_boundary_distance_.cols &&
          static_cast<int>(cam_->h) == img_boundary_distance_.rows);
  }

  elapsed_time_.push_back(
      std::pair<std::string, float>("number of downsampled scene points (HV)", scene_cloud_downsampled_->size()));

  if (occlusion_clouds_.empty()) {  // we can treat single-view as multi-view case with just one view
    if (scene_w_normals_->isOrganized())
      occlusion_clouds_.push_back(scene_w_normals_);
    else {
      // EASY_BLOCK("Depth buffering of scene");
      ScopeTime t("Input point cloud of scene is not organized. Doing depth-buffering to get organized point cloud");
      ZBuffering<PointTWithNormal> zbuf(cam_);
      auto organized_cloud = zbuf.renderPointCloud(*scene_w_normals_);
      occlusion_clouds_.push_back(organized_cloud);
    }

    absolute_camera_poses_.push_back(Eigen::Matrix4f::Identity());
  }

  std::vector<HVRecognitionModel<PointT> *> flat_hypotheses_list;
  for (size_t i = 0, n = obj_hypotheses_groups_.size(); i < n; ++i) {
    for (size_t j = 0, m = obj_hypotheses_groups_[i].size(); j < m; ++j) {
      flat_hypotheses_list.emplace_back(obj_hypotheses_groups_[i][j].get());
    }
  }
  const size_t n_flat_hypotheses = flat_hypotheses_list.size();

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < n_flat_hypotheses; ++i) {
    HVRecognitionModel<PointT> &rm = *flat_hypotheses_list[i];
    computeVisibleModelPoints(rm);
    if (param_.icp_iterations_) {
      if (param_.icp_use_point_to_plane_) {
        if (param_.icp_use_generalized_point_to_plane_) {
          refinePose<pcl::GeneralizedIterativeClosestPoint<PointTWithNormal, PointTWithNormal>>(rm);
        } else {
          refinePose<pcl::IterativeClosestPointWithNormals<PointTWithNormal, PointTWithNormal>>(rm);
        }
      } else
        refinePose<pcl::IterativeClosestPoint<PointTWithNormal, PointTWithNormal>>(rm);
    }
  }

  removeRedundantPoses();

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < n_flat_hypotheses; ++i) {
    HVRecognitionModel<PointT> &rm = *flat_hypotheses_list[i];
    if (rm.isRejected())
      continue;

    if (param_.reject_under_floor_ && transform_to_world_) {
      if (checkIfModelIsUnderFloor(rm, transform_to_world_.get())) {
        VLOG(1) << "Removed " << rm.oh_->model_id_
                << " hypothesis: model part is found under floor by thresh=" << param_.floor_z_min_;
        continue;
      }
    }

    if (param_.icp_iterations_ && param_.recompute_visible_points_after_icp_) {
      computeVisibleModelPoints(rm);
    }

    rm.processSilhouette(param_.do_smoothing_, param_.smoothing_radius_, param_.do_erosion_, param_.erosion_radius_,
                         static_cast<int>(cam_->w));

    computeVisibleOctreeNodes(rm);
    if (param_.check_smooth_clusters_)
      rm.on_smooth_cluster_ = boost::dynamic_bitset<>(max_smooth_label_id_, 0);

    float visible_ratio = (float)rm.visible_indices_by_octree_.size() / (float)rm.num_pts_full_model_;
    if (visible_ratio < param_.min_visible_ratio_) {
      rm.rejected_due_to_low_visibility_ = true;
      VLOG(1) << "Removed " << rm.oh_->model_id_ << " due to low visibility (" << visible_ratio << " by thresh "
              << param_.min_visible_ratio_ << ")!";
      continue;
    }

    if (!param_.ignore_color_even_if_exists_) {
      ScopeTime t("Converting model color values");
      // EASY_BLOCK("Converting model color values");
      convertColor(*rm.visible_cloud_, rm.pt_color_, CV_RGB2Lab);
    }

    computeModelFitness(rm);
  }

  if (vis_model_) {
    for (size_t i = 0; i < obj_hypotheses_groups_.size(); i++) {
      for (size_t jj = 0; jj < obj_hypotheses_groups_[i].size(); jj++) {
        auto rm = obj_hypotheses_groups_[i][jj];
        if (!rm->isRejected()) {
          VLOG(1) << "Visualizing hypothesis element " << jj << " from group " << i << "with id " << rm->oh_->model_id_;
          vis_model_->visualize(this, *rm);
        }
      }
    }
  }

  // do non-maxima surpression on all hypotheses in a hypotheses group based on model fitness (i.e. select only the
  // one hypothesis in group with best model fit)
  global_hypotheses_.resize(obj_hypotheses_groups_.size());

  size_t kept = 0;
  for (size_t i = 0; i < obj_hypotheses_groups_.size(); i++) {
    std::vector<typename HVRecognitionModel<PointT>::Ptr> ohg = obj_hypotheses_groups_[i];
    // remove all rejected hypothesis
    auto it = std::remove_if(ohg.begin(), ohg.end(), [](const auto &hypothesis) { return hypothesis->isRejected(); });
    it = std::max_element(ohg.begin(), it, [](const auto &a, const auto &b) { return a->model_fit_ < b->model_fit_; });
    if (it != ohg.end()) {
      global_hypotheses_[kept++] = *it;
    }
  }
  global_hypotheses_.resize(kept);
  obj_hypotheses_groups_.clear();  // free space

  size_t kept_hypotheses = 0;
  for (size_t i = 0; i < global_hypotheses_.size(); i++) {
    const typename HVRecognitionModel<PointT>::Ptr rm = global_hypotheses_[i];

    rm->is_outlier_ = isOutlier(*rm);

    if (rm->is_outlier_)
      VLOG(1) << rm->oh_->class_id_ << " " << rm->oh_->model_id_ << " is rejected due to low model fitness score.";
    else
      global_hypotheses_[kept_hypotheses++] = global_hypotheses_[i];
  }

  global_hypotheses_.resize(kept_hypotheses);

  if (param_.check_smooth_clusters_) {
    kept_hypotheses = 0;
    ScopeTime t("Computing smooth region intersection");
    computeSmoothRegionOverlap();

    for (size_t i = 0; i < global_hypotheses_.size(); i++) {
      HVRecognitionModel<PointT> &rm = *global_hypotheses_[i];

      if (!rm.isRejected() && rm.violates_smooth_cluster_check_ && smooth_region_overlap_.row(i).sum() == 0) {
        rm.rejected_due_to_smooth_cluster_violation = true;
        VLOG(1) << rm.oh_->class_id_ << " " << rm.oh_->model_id_ << " with hypothesis id " << i
                << " rejected due to smooth cluster violation.";
      } else {
        global_hypotheses_[kept_hypotheses++] = global_hypotheses_[i];
      }
    }
    global_hypotheses_.resize(kept_hypotheses);
  }

  elapsed_time_.push_back(std::pair<std::string, float>("hypotheses left for global optimization", kept_hypotheses));

  if (!kept_hypotheses)
    return;

  {
    ScopeTime t("Computing pairwise intersection");
    computePairwiseIntersection();
  }

  if (vis_pairwise_)
    vis_pairwise_->visualize(this);
}  // namespace v4r

template <typename PointT>
void HypothesisVerification<PointT>::optimize() {
  if (VLOG_IS_ON(1)) {
    VLOG(1) << global_hypotheses_.size()
            << " hypotheses are left for global verification after individual hypotheses "
               "rejection. These are the left hypotheses: ";
    for (size_t i = 0; i < global_hypotheses_.size(); i++)
      VLOG(1) << i << ": " << global_hypotheses_[i]->oh_->class_id_ << " " << global_hypotheses_[i]->oh_->model_id_;
  }

  evaluated_solutions_.clear();
  num_evaluations_ = 0;
  best_solution_.solution_ = boost::dynamic_bitset<>(global_hypotheses_.size(), 0);
  best_solution_.cost_ = std::numeric_limits<double>::max();
  search();

  for (size_t i = 0; i < global_hypotheses_.size(); i++) {
    if (best_solution_.solution_[i])
      global_hypotheses_[i]->oh_->is_verified_ = true;
  }

  std::stringstream info;
  info << "*****************************" << std::endl
       << "Solution: " << best_solution_.solution_ << std::endl
       << "Final cost: " << best_solution_.cost_ << std::endl
       << "Number of evaluations: " << num_evaluations_ << std::endl
       << "*****************************" << std::endl;
  VLOG(1) << info.str();
}

template <typename PointT>
void HypothesisVerification<PointT>::setModelDatabase(const typename Source<PointT>::ConstPtr &m_db) {
  m_db_ = m_db;

  octree_model_representation_.clear();

  // EASY_BLOCK("Computing octrees for model visibility computation");
  ScopeTime t("Computing octrees for model visibility computation");
  const auto models = m_db_->getModels();
  for (const auto &m : models) {
    const auto model_cloud = m->getAssembled();

    auto model_octree_it = octree_model_representation_.find(m->id_);
    if (model_octree_it == octree_model_representation_.end()) {
      std::shared_ptr<pcl::octree::OctreePointCloudPointVector<PointTWithNormal>> octree(
          new pcl::octree::OctreePointCloudPointVector<PointTWithNormal>(param_.octree_resolution_m_));
      octree->setInputCloud(model_cloud);
      octree->addPointsFromInputCloud();
      octree_model_representation_[m->id_] = octree;
    }
  }
}

template <typename PointT>
void HypothesisVerification<PointT>::setSceneCloud(
    const typename pcl::PointCloud<PointTWithNormal>::ConstPtr &scene_cloud) {
  scene_w_normals_ = scene_cloud;
  checkInput();
  downsampleSceneCloud();

#pragma omp parallel sections
  {
#pragma omp section
    {
      // EASY_BLOCK("Computing octree");
      ScopeTime t("Computing octree");
      octree_scene_downsampled_.reset(
          new pcl::octree::OctreePointCloudSearch<PointTWithNormal>(param_.octree_resolution_m_));
      octree_scene_downsampled_->setInputCloud(scene_cloud_downsampled_);
      octree_scene_downsampled_->addPointsFromInputCloud();
    }
#pragma omp section
    {
      if (param_.check_smooth_clusters_) {
        ScopeTime t("Extracting smooth clusters");
        extractEuclideanClustersSmooth();
      }
    }
#pragma omp section
    {
      if (!param_.ignore_color_even_if_exists_) {
        ScopeTime t("Converting scene color values");
        // EASY_BLOCK("Converting scene color values");
        convertColor(*scene_cloud_downsampled_, scene_color_channels_, CV_RGB2Lab);
      }
    }
#pragma omp section
    {
      if (param_.outline_verification_) {
        ScopeTime t("Precompute scene distance field");
        // EASY_BLOCK("Precompute scene distance fields");
        outline_verification_.setScene(*scene_cloud, depth_outlines_param_);
      }
    }
  }
}

template <typename PointT>
void HypothesisVerification<PointT>::checkInput() const {
  if (scene_w_normals_->isOrganized()) {
    // check if camera calibration file is for the same image resolution as depth image
    if (cam_->w != scene_w_normals_->width || cam_->h != scene_w_normals_->height) {
      LOG(ERROR) << "Input cloud has different resolution (" << scene_w_normals_->width << "x"
                 << scene_w_normals_->height << ") than resolution stated in camera calibration file (" << cam_->w
                 << "x" << cam_->h << ").";
    }
  } else {
    if (cam_->w != 640 || cam_->h != 480) {
      LOG(INFO) << "Input cloud is not organized(" << scene_w_normals_->width << "x" << scene_w_normals_->height
                << ") and resolution stated in camera calibration file (" << cam_->w << "x" << cam_->h
                << ") is not VGA! Please check if this is okay.";
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
void HypothesisVerification<PointT>::verify() {
  elapsed_time_.clear();

  size_t num_hypotheses = 0;
  for (size_t i = 0; i < obj_hypotheses_groups_.size(); i++)
    num_hypotheses += obj_hypotheses_groups_[i].size();

  elapsed_time_.push_back(std::pair<std::string, float>("number of hypotheses", num_hypotheses));

  if (!num_hypotheses) {
    LOG(INFO) << "No hypotheses to verify!";
    return;
  }

  {
    ScopeTime t("Computing cues for object hypotheses verification");
    initialize();
  }

  {
    ScopeTime t("Optimizing object hypotheses verification cost function");
    optimize();
  }

  cleanUp();
}

template <typename PointT>
void HypothesisVerification<PointT>::computeModelFitness(HVRecognitionModel<PointT> &rm) const {
  // EASY_BLOCK("compute model fitness");
  //    pcl::visualization::PCLVisualizer vis;
  //    int vp1, vp2;
  //    vis.createViewPort(0,0,0.5,1,vp1);
  //    vis.createViewPort(0.5,0,1,1,vp2);
  //    vis.addPointCloud(rm.visible_cloud_, "vis_cloud", vp1);
  //    pcl::visualization::PointCloudColorHandlerCustom<PointT> gray (scene_cloud_downsampled_, 128, 128, 128);
  //    vis.addPointCloud(scene_cloud_downsampled_, gray, "scene1", vp1);
  //    vis.setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "scene1");

  for (const auto &image_mask : rm.image_mask_) {
    if (outline_verification_.isOutlier(image_mask)) {
      VLOG(1) << "Hypothesis rejected due to model<>scene outlines mismatch";
      rm.model_fit_ = 0.f;
      rm.oh_->confidence_ = 0.f;
      rm.rejected_due_to_outline_mismatch_ = true;
      return;
    }
  }

  std::vector<float> nn_sqrd_distances;
  std::vector<int> nn_indices;
  for (size_t midx = 0; midx < rm.visible_cloud_->size(); midx++) {
    PointTWithNormal query_pt;
    query_pt.getVector4fMap() = rm.visible_cloud_->at(midx).getVector4fMap();
    octree_scene_downsampled_->radiusSearch(query_pt, param_.inlier_threshold_xyz_, nn_indices, nn_sqrd_distances);

    //        vis.addSphere(rm.visible_cloud_->points[midx], 0.005, 0., 1., 0., "queryPoint", vp1 );

    const auto normal_m = rm.visible_cloud_->points[midx].getNormalVector4fMap();

    for (size_t k = 0; k < nn_indices.size(); k++) {
      int sidx = nn_indices[k];
      //            std::stringstream pt_id; pt_id << "searched_pt_" << k;
      //            vis.addSphere(scene_cloud_downsampled_->points[sidx], 0.005, 1., 0., 0., pt_id.str(), vp2 );
      //            vis.addPointCloud(rm.visible_cloud_, "vis_cloud2", vp2);
      //            vis.addPointCloud(scene_cloud_downsampled_, gray, "scene2", vp2);
      //            vis.setPointCloudRenderingProperties( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.2, "scene2");

      ModelSceneCorrespondence c(sidx, midx);
      const auto normal_s = scene_cloud_downsampled_->at(sidx).getNormalVector4fMap();
      c.normals_dotp_ = normal_m.dot(normal_s);

      if (!param_.ignore_color_even_if_exists_) {
        const auto &color_m = rm.pt_color_[midx];
        const auto &color_s = scene_color_channels_[sidx];
        c.color_distance_ = color_dist_f_(lab_conversion(color_s), lab_conversion(color_m));
      }

      c.fitness_ = getFitness(c);
      rm.model_scene_c_.push_back(c);
    }
    //        vis.removeAllShapes(vp1);
  }

  //    vis.spin();
  //    rm.model_scene_c_.resize(kept);

  std::sort(rm.model_scene_c_.begin(), rm.model_scene_c_.end());

  if (param_.use_histogram_specification_) {
    boost::dynamic_bitset<> scene_pt_is_taken(scene_cloud_downsampled_->size(), 0);
    std::vector<float> scene_l_vals(scene_cloud_downsampled_->size());

    size_t kept = 0;
    for (const ModelSceneCorrespondence &c : rm.model_scene_c_) {
      size_t sidx = c.scene_id_;
      if (!scene_pt_is_taken[sidx]) {
        scene_pt_is_taken.set(sidx);
        scene_l_vals[kept++] = static_cast<float>(scene_color_channels_[sidx](0));
      }
    }
    scene_l_vals.resize(kept);

    if (kept) {
      float mean_l_value_scene = accumulate(scene_l_vals.begin(), scene_l_vals.end(), 0.f) / scene_l_vals.size();
      float mean_l_value_model = accumulate(rm.pt_color_.begin(), rm.pt_color_.end(), 0.f,
                                            [](const auto &sum, const auto &v) { return sum + v[0]; }) /
                                 rm.pt_color_.size();

      float max_l_offset = 15.f;
      float l_compensation =
          std::max<float>(-max_l_offset, std::min<float>(max_l_offset, (mean_l_value_scene - mean_l_value_model)));
      std::for_each(rm.pt_color_.begin(), rm.pt_color_.end(), [l_compensation](auto &c) { c[0] += l_compensation; });

      for (ModelSceneCorrespondence &c : rm.model_scene_c_) {
        size_t sidx = c.scene_id_;
        size_t midx = c.model_id_;

        const auto &color_m = rm.pt_color_[midx];
        const auto &color_s = scene_color_channels_[sidx];
        c.color_distance_ = color_dist_f_(lab_conversion(color_s), lab_conversion(color_m));
        c.fitness_ = getFitness(c);
      }
    }
  }

  Eigen::Array<bool, Eigen::Dynamic, 1> scene_explained_pts(scene_cloud_downsampled_->size());
  scene_explained_pts.setConstant(scene_cloud_downsampled_->size(), false);

  Eigen::Array<bool, Eigen::Dynamic, 1> model_explained_pts(rm.visible_cloud_->size());
  model_explained_pts.setConstant(rm.visible_cloud_->size(), false);

  Eigen::VectorXf modelFit = Eigen::VectorXf::Zero(rm.visible_cloud_->size());
  rm.scene_explained_weight_ = Eigen::SparseVector<float>(scene_cloud_downsampled_->size());
  rm.scene_explained_weight_.reserve(rm.model_scene_c_.size());

  for (const ModelSceneCorrespondence &c : rm.model_scene_c_) {
    size_t sidx = c.scene_id_;
    size_t midx = c.model_id_;

    if (!scene_explained_pts(sidx)) {
      scene_explained_pts(sidx) = true;
      rm.scene_explained_weight_.insert(sidx) = c.fitness_;
    }

    if (!model_explained_pts(midx)) {
      model_explained_pts(midx) = true;
      modelFit(midx) = c.fitness_;
    }
  }

  if (param_.check_smooth_clusters_) {
    // save which smooth clusters align with hypotheses
    for (size_t i = 1; i < max_smooth_label_id_;
         i++) {  // ignore label 0 as these are unlabeled clusters (which are e.g.
      // smaller than the minimal required cluster size)
      Eigen::Array<bool, Eigen::Dynamic, 1> s_pt_in_region = (scene_pt_smooth_label_id_.array() == i);
      Eigen::Array<bool, Eigen::Dynamic, 1> explained_pt_in_region =
          (s_pt_in_region.array() && scene_explained_pts.array());
      size_t num_explained_pts_in_region = explained_pt_in_region.count();
      size_t num_pts_in_smooth_regions = s_pt_in_region.count();

      rm.on_smooth_cluster_[i] = num_explained_pts_in_region > 0;

      if (num_explained_pts_in_region > param_.min_pts_smooth_cluster_to_be_epxlained_ &&
          (float)(num_explained_pts_in_region) / num_pts_in_smooth_regions < param_.min_ratio_cluster_explained_) {
        rm.violates_smooth_cluster_check_ = true;
      }
    }
  }

  rm.model_fit_ = modelFit.sum();
  rm.oh_->confidence_ = rm.visible_cloud_->empty() ? 0.f : rm.model_fit_ / rm.visible_cloud_->size();

  VLOG(1) << "model fit of " << rm.oh_->model_id_ << ": " << rm.model_fit_
          << " (normalized: " << rm.model_fit_ / rm.visible_cloud_->size() << ").";
}

// template class  HypothesisVerification<pcl::PointXYZ>;
template class  HypothesisVerification<pcl::PointXYZRGB>;
}  // namespace v4r
