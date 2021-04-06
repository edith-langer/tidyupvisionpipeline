#include <glog/logging.h>
#include <v4r/common/point_types.h>
#include <v4r/segmentation/plane_extractor.h>
#include <pcl/impl/instantiate.hpp>

namespace po = boost::program_options;

namespace v4r {
void PlaneExtractorParameter::init(boost::program_options::options_description &desc, const std::string &section_name) {
  desc.add_options()((section_name + ".max_iterations").c_str(),
                     po::value<size_t>(&max_iterations_)->default_value(max_iterations_),
                     "maximum number of iterations the sample consensus method will run");
  desc.add_options()((section_name + ".min_num_plane_inliers").c_str(),
                     po::value<size_t>(&min_num_plane_inliers_)->default_value(min_num_plane_inliers_),
                     "minimum number of plane inliers");
  desc.add_options()((section_name + ".distance_threshold").c_str(),
                     po::value<double>(&distance_threshold_)->default_value(distance_threshold_),
                     "tolerance in meters for difference in perpendicular distance (d component "
                     "of plane equation) to the plane between neighboring points, to be "
                     "considered part of the same plane");
  desc.add_options()((section_name + ".compute_all_planes").c_str(),
                     po::value<bool>(&compute_all_planes_)->default_value(compute_all_planes_),
                     "if true, computes all planes (also if method does not compute all of them intrinsically)");
  desc.add_options()((section_name + ".optimize_cofficients").c_str(),
                     po::value<bool>(&optimize_cofficients_)->default_value(optimize_cofficients_),
                     "if true, enables model coefficient refinement");
  desc.add_options()((section_name + ".eps_angle").c_str(), po::value<double>(&eps_angle_)->default_value(eps_angle_),
                     "maximum allowed difference between the plane normal and the given axis");
  desc.add_options()((section_name + ".model_type").c_str(), po::value<int>(&model_type_)->default_value(model_type_),
                     "Model type used for SAmple Consensus (SAC)");
  desc.add_options()((section_name + ".method_type").c_str(),
                     po::value<int>(&method_type_)->default_value(method_type_),
                     "Method type used for SAmple Consensus (SAC)");
  desc.add_options()((section_name + ".probability").c_str(),
                     po::value<double>(&probability_)->default_value(probability_),
                     "Probability used for SAmple Consensus (SAC)");
  desc.add_options()((section_name + ".samples_max_distance").c_str(),
                     po::value<double>(&samples_max_distance_)->default_value(samples_max_distance_),
                     "maximum distance in meter allowed when drawing random samples for SAmple Consensus (SAC)");
  desc.add_options()(
      (section_name + ".z_adaptive").c_str(), po::value<bool>(&z_adaptive_)->default_value(z_adaptive_),
      "if true, scales the distance threshold parameter for organized multiplane detection linear with distance");
  desc.add_options()((section_name + ".check_normals").c_str(),
                     po::value<bool>(&check_normals_)->default_value(check_normals_),
                     "if true, discards points that are on the plane normal but have a surface normal orientation that "
                     "is different to the plane's surface normal orientation by eps_angle_");

  // Parameters for Tile Plane Extractor
  desc.add_options()((section_name + ".minNrPatches").c_str(),
                     po::value<size_t>(&minNrPatches_)->default_value(minNrPatches_),
                     "The minimum number of blocks that are allowed to spawn a plane");
  desc.add_options()((section_name + ".patchDim").c_str(), po::value<size_t>(&patchDim_)->default_value(patchDim_),
                     "The minimum number of blocks that are allowed to spawn a plane");
  desc.add_options()((section_name + ".minBlockInlierRatio").c_str(),
                     po::value<float>(&minBlockInlierRatio_)->default_value(minBlockInlierRatio_),
                     "The minimum ratio of points that have to be in a patch before it would get discarded");
  desc.add_options()((section_name + ".pointwiseNormalCheck").c_str(),
                     po::value<bool>(&pointwiseNormalCheck_)->default_value(pointwiseNormalCheck_),
                     "Activating this allows to reduce a lot of calculations and improves speed.");
  desc.add_options()((section_name + ".useVariableThresholds").c_str(),
                     po::value<bool>(&useVariableThresholds_)->default_value(useVariableThresholds_),
                     " useVariableThresholds");
  desc.add_options()((section_name + ".maxInlierBlockDist").c_str(),
                     po::value<float>(&maxInlierBlockDist_)->default_value(maxInlierBlockDist_),
                     "The maximum distance two adjacent patches are allowed to be out of plane");
  desc.add_options()((section_name + ".doZTest").c_str(), po::value<bool>(&doZTest_)->default_value(doZTest_),
                     "Only the closest possible points get added to a plane");
  desc.add_options()((section_name + ".use_variable_thresholds").c_str(),
                     po::value<bool>(&useVariableThresholds_)->default_value(useVariableThresholds_),
                     "Uses z-adaptive inlier threshold");
}

template <typename PointT>
std::vector<std::vector<int>> PlaneExtractor<PointT>::getPlaneInliers() {
  if (plane_inliers_.size() != all_planes_.size()) {
    float eps_angle_dotp_threshold = cos(param_.eps_angle_);
    plane_inliers_.resize(all_planes_.size());

    for (size_t i = 0; i < all_planes_.size(); i++) {
      for (size_t pt_id = 0; pt_id < cloud_->size(); pt_id++) {
        if (fabs(cloud_->points[pt_id].getVector4fMap().dot(all_planes_[i])) < param_.distance_threshold_) {
          if (normal_cloud_ &&
              param_.check_normals_) {  // check if plane orientation and surface normal of point approximately align
            const pcl::Normal &pt_n = normal_cloud_->points[pt_id];

            if (!pcl::isFinite(pt_n))
              continue;

            float angle_dotp =
                all_planes_[i].dot(pt_n.getNormalVector4fMap());  // Note: last element of point normal is
            // always zero and plane equation
            // normalized on the first three
            // coefficients (surface normal of plane)
            if (fabs(angle_dotp) < eps_angle_dotp_threshold)
              continue;
          }
          plane_inliers_[i].push_back(pt_id);
        }
      }
    }
  }
  return plane_inliers_;
}

template <typename PointT>
void PlaneExtractor<PointT>::compute(const boost::optional<const Eigen::Vector3f> &search_axis) {
  CHECK(!getRequiresNormals() || normal_cloud_) << "Plane extractor requires normals, but they were not provided";

  do_compute(search_axis);

  for (auto &plane_eq : all_planes_) {
    // flip plane orientation towards viewpoint
    if (plane_eq.dot(Eigen::Vector4f::UnitZ()) > 0.f) {
      plane_eq *= -1.f;
    }

    // normalize plane normal vector
    plane_eq /= plane_eq.head(3).norm();
  }

  if (search_axis && (param_.method_type_ == pcl::SACMODEL_PERPENDICULAR_PLANE ||
                      param_.method_type_ == pcl::SACMODEL_PARALLEL_PLANE)) {
    // check if plane normal aligns with search axis
    const float eps_angle_dotp_threshold = cos(param_.eps_angle_);
    const float eps_angle_dotp_threshold_perpendicular = cos(M_PI / 2 - param_.eps_angle_);

    size_t kept = 0;
    for (size_t i = 0; i < all_planes_.size(); i++) {
      const float angle_dotp = all_planes_[i].head(3).dot(search_axis.get());

      bool is_kept = false;
      if (param_.method_type_ == pcl::SACMODEL_PARALLEL_PLANE) {
        if (fabs(angle_dotp) > eps_angle_dotp_threshold) {
          is_kept = true;
        }
      }

      if (param_.method_type_ == pcl::SACMODEL_PERPENDICULAR_PLANE) {
        if (fabs(angle_dotp) < eps_angle_dotp_threshold_perpendicular) {
          is_kept = true;
        }
      }

      if (is_kept) {
        all_planes_[kept] = all_planes_[i];
        if (plane_inliers_.size() > i) {
          plane_inliers_[kept] = plane_inliers_[i];
        }
        kept++;
      }
    }
    all_planes_.resize(kept);
    if (plane_inliers_.size() > kept) {
      plane_inliers_.resize(kept);
    }
  }
}

#define PCL_INSTANTIATE_PlaneExtractor(T) template class PlaneExtractor<T>;
PCL_INSTANTIATE(PlaneExtractor, V4R_PCL_XYZ_POINT_TYPES)

}  // namespace v4r
