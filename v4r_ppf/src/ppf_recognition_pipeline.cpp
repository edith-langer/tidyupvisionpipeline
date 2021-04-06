#include <algorithm>
#include <numeric>

#include <boost/format.hpp>

#include <glog/logging.h>

#include <v4r/common/downsampler.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/geometry/average.h>
#include <ppf/correspondence.h>
#include <ppf_recognition_pipeline.h>
#include <ppf/impl/correspondence_finder.hpp>

#include <pcl/common/angles.h>
#include <pcl/io/pcd_io.h>

namespace po = boost::program_options;

// Anonymous namespace with local helper functions
namespace {

struct PosesWithScore {
  std::vector<Eigen::Affine3f, Eigen::aligned_allocator<Eigen::Affine3f>> poses;
  float score = 0.0f;
};

// Cluster correspondences (equivalently, pose hypotheses).
// If the symmetry_rotations is not empty, takes into account rotational equivalence while clustering.
std::vector<PosesWithScore> clusterCorrespondences(const ppf::Correspondence::Vector& correspondences,
                                                   float distance_threshold, float angle_threshold,
                                                   const std::vector<Eigen::Matrix3f>& symmetry_rotations) {
  // Sort correspondences by the weight to make sure that most likely poses are at the core of created clusters.
  std::vector<size_t> indices(correspondences.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&](size_t i1, size_t i2) { return correspondences[i1].weight > correspondences[i2].weight; });

  // Struct to represent a cluster of correspondences.
  struct Cluster {
    PosesWithScore ps;
    std::vector<Eigen::Matrix3f> rotations;
  };
  std::vector<Cluster> clusters;

  // Create a vector of rotations that includes identity and all passed symmetry rotations.
  std::vector<Eigen::Matrix3f> rotations = {Eigen::Matrix3f::Identity()};
  rotations.insert(rotations.end(), symmetry_rotations.begin(), symmetry_rotations.end());

  for (const auto& index : indices) {
    const auto& weight = correspondences.at(index).weight;
    const auto& pose = correspondences.at(index).pose;
    bool found_cluster = false;
    for (auto& cluster : clusters) {
      const float position_diff = (pose.translation() - cluster.ps.poses.front().translation()).norm();
      if (position_diff > distance_threshold)
        continue;

      for (size_t i = 0; i < cluster.rotations.size(); ++i) {
        const auto& R = cluster.rotations[i];
        const Eigen::AngleAxisf rotation_diff_mat((R.lazyProduct(pose.rotation()).eval()));
        const float rotation_diff_angle = std::abs(rotation_diff_mat.angle());
        if (rotation_diff_angle < angle_threshold) {
          cluster.ps.score += weight;
          cluster.ps.poses.push_back(pose);
          cluster.ps.poses.back().linear() = cluster.ps.poses.back().linear() * rotations[i];
          found_cluster = true;
          break;
        }
      }

      if (found_cluster)
        break;
    }
    if (!found_cluster) {
      Cluster cluster;
      cluster.ps.poses.push_back(pose);
      cluster.ps.score = weight;
      for (const auto& R : rotations)
        cluster.rotations.push_back((pose.rotation() * R).inverse());
      clusters.push_back(std::move(cluster));
    }
  }

  std::vector<PosesWithScore> results;
  results.reserve(clusters.size());
  std::transform(clusters.begin(), clusters.end(), std::back_inserter(results),
                 [](auto& c) { return std::move(c.ps); });
  return results;
}

// Given a symmetry descriptor, reorder point cloud such that the points that are located on the positive side of
// each of the symmetry planes are listed first in the point cloud. Also count the number of such points since this
// is a needed information for ModelSearch.
template <typename PointT>
pcl::PointCloud<PointT> reorderPointCloud(const pcl::PointCloud<PointT>& cloud, const std::vector<bool>& symmetry_xyz,
                                          size_t& num_anchors) {
  std::vector<Eigen::Vector3f> planes;
  if (symmetry_xyz[0])
    planes.push_back(Eigen::Vector3f::UnitZ());
  if (symmetry_xyz[1])
    planes.push_back(Eigen::Vector3f::UnitX());
  if (symmetry_xyz[2])
    planes.push_back(Eigen::Vector3f::UnitY());
  pcl::PointCloud<PointT> reordered(cloud.size(), 1);
  auto anchors = reordered.begin();
  auto rest = reordered.end() - 1;
  for (const auto& point : cloud)
    for (const auto& plane : planes)
      if (point.getVector3fMap().dot(plane) > 0)
        *anchors++ = point;
      else
        *rest-- = point;
  num_anchors = std::distance(reordered.begin(), anchors);
  return reordered;
}

std::vector<Eigen::Matrix3f> computeSymmetryRotations(const std::vector<bool>& symmetry_xyz) {
  CHECK(symmetry_xyz.size() == 3) << "Invalid symmetry XYZ descriptor, should have 3 booleans";
  std::vector<Eigen::Matrix3f> rotations;
  if (symmetry_xyz[0])
    rotations.push_back(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX()).matrix());
  if (symmetry_xyz[1])
    rotations.push_back(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitY()).matrix());
  if (symmetry_xyz[2])
    rotations.push_back(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitZ()).matrix());
  return rotations;
}

}  // anonymous namespace

namespace v4r {

void PPFRecognitionPipelineParameter::init(boost::program_options::options_description& desc,
                                           const std::string& section_name) {
  desc.add_options()((section_name + ".downsampling_resolution").c_str(),
                     po::value<float>(&downsampling_resolution_)->default_value(downsampling_resolution_),
                     "Downsampling resolution for model and scene point clouds (fraction of model diameter)");
  desc.add_options()((section_name + ".ppf_distance_quantization_step").c_str(),
                     po::value<float>(&ppf_distance_quantization_step_)->default_value(ppf_distance_quantization_step_),
                     "Quantization step for distances in PPF (fraction of model diameter)");
  desc.add_options()((section_name + ".ppf_angle_quantization_step").c_str(),
                     po::value<float>(&ppf_angle_quantization_step_)->default_value(ppf_angle_quantization_step_),
                     "Quantization step for angles in PPF (degrees)");
  desc.add_options()((section_name + ".scene_subsampling_rate").c_str(),
                     po::value<size_t>(&scene_subsampling_rate_)->default_value(scene_subsampling_rate_),
                     "Pose hypotheses are generated only for every n-th point in the scene");
  desc.add_options()(
      (section_name + ".correspondences_per_scene_point").c_str(),
      po::value<size_t>(&correspondences_per_scene_point_)->default_value(correspondences_per_scene_point_),
      "Number of correspondences generated per scene point");
  desc.add_options()((section_name + ".min_votes").c_str(), po::value<size_t>(&min_votes_)->default_value(min_votes_),
                     "Minimum required number of votes in Hough Voting scheme");
  desc.add_options()((section_name + ".max_hypotheses").c_str(),
                     po::value<size_t>(&max_hypotheses_)->default_value(max_hypotheses_),
                     "Maximum number of pose hypotheses to output");
  desc.add_options()(
      (section_name + ".pose_clustering_distance_threshold").c_str(),
      po::value<float>(&pose_clustering_distance_threshold_)->default_value(pose_clustering_distance_threshold_),
      "Distance threshold for clustering together pose hypotheses (fraction of model diameter)");
  desc.add_options()(
      (section_name + ".pose_clustering_angle_threshold").c_str(),
      po::value<float>(&pose_clustering_angle_threshold_)->default_value(pose_clustering_angle_threshold_),
      "Angular threshold for clustering together pose hypotheses (degrees)");
  desc.add_options()((section_name + ".no_use_symmetry").c_str(),
                     po::bool_switch()->notifier([this](bool v) { this->use_symmetry_ = !v; }),
                     "Do not use symmetry information");
  desc.add_options()((section_name + ".use_color").c_str(),
                     po::bool_switch()->notifier([this](bool v) {this->use_color_ = v;}),
                    "If set, uses color in addition to compute the hash.");
  desc.add_options()((section_name + ".check_col_before_voting").c_str(),
                     po::bool_switch()->notifier([this](bool v) {this->check_col_before_voting_ = v;}),
                    "If set, compares color between scene and model point pair before casting a vote.");
  desc.add_options()((section_name + ".ppf_color_hue_quantization_step").c_str(),
                     po::value<float>(&color_quantization_step_[0])->default_value(color_quantization_step_[0])
          ->notifier([](float value)
                {
                  if (value < 0.0f || value > 1.0f) {
                    throw "Color quantization step for hue has to be in the range [0,1]";
                  }
                }),
                     "Quantization step for color in PPF (between 0 and 1)");
  desc.add_options()((section_name + ".ppf_color_saturation_quantization_step").c_str(),
                     po::value<float>(&color_quantization_step_[1])->default_value(color_quantization_step_[1])
          ->notifier([](float value)
                {
                  if (value < 0.0f || value > 1.0f) {
                    throw "Color quantization step for saturation has to be in the range [0,1]";
                  }
                }),
                     "Quantization step for color in PPF (between 0 and 1)");
  desc.add_options()((section_name + ".ppf_color_value_quantization_step").c_str(),
                     po::value<float>(&color_quantization_step_[2])->default_value(color_quantization_step_[2])
          ->notifier([](float value)
                {
                  if (value < 0.0f || value > 1.0f) {
                    throw "Color quantization step for brightness has to be in the range [0,1]";
                  }
                }),
                     "Quantization step for color in PPF (between 0 and 1)");
  desc.add_options()(
      (section_name + ".inlier_threshold_color").c_str(),
      po::value<float>(&inlier_threshold_color_)
          ->default_value(inlier_threshold_color_, boost::str(boost::format("%.2e") % inlier_threshold_color_)),
      "allowed chrominance (AB channel of LAB color space) variance for a point of an object hypotheses to be "
      "considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get "
      "rejected)");
}

template <typename PointT>
void PPFRecognitionPipeline<PointT>::doInit(const bf::path& trained_dir, bool force_retrain,
                                            const std::vector<std::string>& object_instances_to_load) {
    boost::format cache_fmt("ppf_model_d%.0f_a%.0f_ds%0.f_spreading%s%s.hash");

  const auto& models = m_db_->getModels();
  for (const auto& m : models) {
    const auto model_name = m->id_;
    if (!object_instances_to_load.empty() && std::find(object_instances_to_load.begin(), object_instances_to_load.end(),
                                                       model_name) == object_instances_to_load.end()) {
      LOG(INFO) << "Skipping object " << m->id_ << " because it is not in the lists of objects to load.";
      continue;
    }

    // Compute absolute downsampling resolution and quantization step based on model diameter.
    const auto diameter = m->getDiameter();
    auto downsampling_resolution = param_.downsampling_resolution_ * diameter;
    auto dqs = param_.ppf_distance_quantization_step_ * diameter;
    auto aqs = pcl::deg2rad(param_.ppf_angle_quantization_step_);
    auto cqs = param_.color_quantization_step_;
    // Does the model have symmetries? Should we use them?
    const auto& sym = m->properties_.symmetry_xyz_;
    bool use_symmetry = param_.use_symmetry_ ? sym[0] | sym[1] | sym[2] : false;
    bool use_color = param_.use_color_;
    // Construct expected cached model file name based on parameters
    auto fn =
        trained_dir / model_name / boost::str(cache_fmt % (dqs * 1e6) % (aqs * 1e6) % (downsampling_resolution * 1e6)  %(use_symmetry ? "_sym" : "") %
                                                         (use_color ? "_color" + std::to_string(cqs[0] * 1e2) + "_" + std::to_string(cqs[1] * 1e2) + "_" + std::to_string(cqs[2] * 1e2) : ""));

    if (bf::is_regular_file(fn) && !force_retrain) {
      // Load "trained" model search from file
      model_search_[model_name].reset(new ppf::ModelSearch(fn.string()));
    } else {
      // "Train" a new model search and cache for future
      DownsamplerParameter param;
      param.method_ = DownsamplerParameter::Method::ADVANCED;
      param.resolution_ = downsampling_resolution;
      auto model_cloud = m->getAssembled(param);
      if (use_symmetry) {
        size_t num_anchors = model_cloud->size();
        auto reordered = reorderPointCloud(*model_cloud, m->properties_.symmetry_xyz_, num_anchors);
        model_search_[model_name].reset(new ppf::ModelSearch(reordered, dqs, aqs, cqs,
               ppf::ModelSearch::Spreading::On, num_anchors, use_color ? ppf::ModelSearch::FeatureType::CPPF : ppf::ModelSearch::FeatureType::PPF));

      } else {
          model_search_[model_name].reset(new ppf::ModelSearch(*model_cloud, dqs, aqs, cqs, ppf::ModelSearch::Spreading::On, 0,
               use_color ? ppf::ModelSearch::FeatureType::CPPF : ppf::ModelSearch::FeatureType::PPF));

      }
      model_search_[model_name]->save(fn.string());
    }
    // Initialize symmetry rotations for the model (if requested and if symmetries are present)
    if (param_.use_symmetry_) {
      symmetry_rotations_[model_name] = computeSymmetryRotations(m->properties_.symmetry_xyz_);
      LOG(INFO) << "Object " << m->id_ << " has " << symmetry_rotations_[model_name].size() << " symmetry rotations";
    }
  }
}

template <typename PointT>
void PPFRecognitionPipeline<PointT>::do_recognize(const std::vector<std::string>& model_ids_to_search) {
  CHECK(scene_) << "Scene point cloud not set";
  CHECK(scene_normals_) << "Scene normals not set";
  CHECK(scene_normals_->size() == scene_->size()) << "Scene normals do not match in size with scene point cloud";

  for (const auto& model_name : model_ids_to_search) {
    const auto m = m_db_->getModelById("", model_name);

    const auto& model_search = model_search_.at(model_name);
    // Not using at() because this will not exist if the user disabled "use_symmetry" option.
    const auto& symmetry_rotations = symmetry_rotations_[model_name];

    DownsamplerParameter param;
    param.method_ = DownsamplerParameter::Method::ADVANCED;  // results are better with advanced downsampling
    param.resolution_ = param_.downsampling_resolution_ * model_search->getModelDiameter();
    v4r::Downsampler downsampler(param);
    typename pcl::PointCloud<PointTWithNormal>::Ptr scene_with_normals(new pcl::PointCloud<PointTWithNormal>);
    pcl::concatenateFields(*scene_, *scene_normals_, *scene_with_normals);
    auto downsampled = downsampler.downsample<PointTWithNormal>(scene_with_normals);

    LOG(INFO) << "Downsampled scene. Num points: " << downsampled->size();

    ppf::CorrespondenceFinder<PointTWithNormal> cfinder;
    cfinder.setInput(downsampled);
    cfinder.setMaxCorrespondences(param_.correspondences_per_scene_point_);
    cfinder.setMinVotes(param_.min_votes_);
    cfinder.setUseColorCheck(param_.check_col_before_voting_, param_.inlier_threshold_color_);
    cfinder.setModelSearch(model_search);

    LOG(INFO) << "Searching for model " << model_name;

    ppf::ModelSearch::FeatureType ppf_type = param_.use_color_ ? ppf::ModelSearch::FeatureType::CPPF : ppf::ModelSearch::FeatureType::PPF;

    ppf::Correspondence::Vector correspondences;
#pragma omp parallel for schedule(dynamic)
    for (size_t scene_reference_index = 0; scene_reference_index < downsampled->size();
         scene_reference_index += param_.scene_subsampling_rate_) {
      const auto& cc = cfinder.find(scene_reference_index, ppf_type);
#pragma omp critical
      correspondences.insert(correspondences.end(), cc.begin(), cc.end());
    }

    LOG(INFO) << "Found " << correspondences.size() << " correspondences to the model";

    auto distance_threshold = param_.pose_clustering_distance_threshold_ * model_search->getModelDiameter();
    auto angle_threshold = pcl::deg2rad(param_.pose_clustering_angle_threshold_);
    auto clusters = clusterCorrespondences(correspondences, distance_threshold, angle_threshold, symmetry_rotations);
    std::sort(clusters.begin(), clusters.end(), [](const auto& a, const auto& b) { return a.score > b.score; });

    LOG(INFO) << "Clustered into " << clusters.size() << " clusters";

    size_t num_clusters = clusters.size();
    if (param_.max_hypotheses_ && num_clusters > param_.max_hypotheses_)
      num_clusters = param_.max_hypotheses_;

    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> transforms(num_clusters);
    for (size_t cluster_id = 0; cluster_id < num_clusters; ++cluster_id)
      transforms[cluster_id] = geometry::averageTransforms<float>(clusters[cluster_id].poses).matrix();

    ObjectHypothesesGroup group;
    for (size_t i = 0; i < num_clusters; ++i) {
      group.ohs_.emplace_back(new ObjectHypothesis);
      group.ohs_.back()->transform_ = transforms[i];
      group.ohs_.back()->model_id_ = model_name;
      group.ohs_.back()->class_id_ = "";
      group.ohs_.back()->confidence_wo_hv_ = clusters[i].score;
      LOG(INFO) << "Hypothesis " << i << ", votes: " << clusters[i].score
                << ", num correspondences: " << clusters[i].poses.size();
    }
    group.global_hypotheses_ = false;
    obj_hypotheses_.push_back(std::move(group));
  }
}

template class PPFRecognitionPipeline<pcl::PointXYZRGB>;

}  // namespace v4r
