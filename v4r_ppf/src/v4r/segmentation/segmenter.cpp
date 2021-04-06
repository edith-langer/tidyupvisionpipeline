#include <v4r/segmentation/segmenter.h>

namespace po = boost::program_options;

namespace v4r {
void SegmenterParameter::init(boost::program_options::options_description &desc, const std::string &section_name) {
  desc.add_options()((section_name + ".min_cluster_size").c_str(),
                     po::value<size_t>(&min_cluster_size_)->default_value(min_cluster_size_),
                     "minimum number of points in a cluster");
  desc.add_options()((section_name + ".max_cluster_size").c_str(),
                     po::value<size_t>(&max_cluster_size_)->default_value(max_cluster_size_), "");
  desc.add_options()(
      (section_name + ".distance_threshold").c_str(),
      po::value<double>(&distance_threshold_)->default_value(distance_threshold_),
      "tolerance in meters for difference in perpendicular distance (d component of plane equation) to the plane "
      "between neighboring points, to be considered part of the same plane");
  desc.add_options()(
      (section_name + ".angular_threshold_deg").c_str(),
      po::value<double>(&angular_threshold_deg_)->default_value(angular_threshold_deg_),
      "tolerance in gradients for difference in normal direction between neighboring points, to be considered part "
      "of the same plane.");
  desc.add_options()((section_name + ".wsize").c_str(), po::value<int>(&wsize_)->default_value(wsize_), "");
  desc.add_options()((section_name + ".z_adaptive").c_str(), po::value<bool>(&z_adaptive_)->default_value(z_adaptive_),
                     "if true, scales the smooth segmentation parameters linear with distance (constant till 1m at the "
                     "given parameters)");
  desc.add_options()((section_name + ".compute_planar_patches_only").c_str(),
                     po::value<bool>(&compute_planar_patches_only_)->default_value(compute_planar_patches_only_),
                     "if true, only compute planar surface patches");
  desc.add_options()((section_name + ".planar_inlier_dist").c_str(),
                     po::value<float>(&planar_inlier_dist_)->default_value(planar_inlier_dist_),
                     "maximum allowed distance of a point to the plane");
  desc.add_options()((section_name + ".octree_resolution").c_str(),
                     po::value<float>(&octree_resolution_)->default_value(octree_resolution_),
                     "octree resolution in meter");
  desc.add_options()((section_name + ".curvature_threshold").c_str(),
                     po::value<float>(&curvature_threshold_)->default_value(curvature_threshold_),
                     "smooth clustering threshold for curvature");
  desc.add_options()((section_name + ".force_unorganized").c_str(),
                     po::value<bool>(&force_unorganized_)->default_value(force_unorganized_),
                     "if true, searches for neighboring points using a kdtree and not exploiting the organized pixel "
                     "structure (even if input cloud is organized)");
  desc.add_options()((section_name + ".downsample_before_segmentation").c_str(),
                     po::value<bool>(&downsample_before_segmentation_)->default_value(downsample_before_segmentation_),
                     "the resolution in meter used for uniform sampling before euclidean clustering");
  desc.add_options()((section_name + ".downsample_before_segmentation_resolution").c_str(),
                     po::value<float>(&downsample_before_segmentation_resolution_)
                         ->default_value(downsample_before_segmentation_resolution_),
                     "the resolution in meter used for uniform sampling before euclidean clustering");
}
}  // namespace v4r