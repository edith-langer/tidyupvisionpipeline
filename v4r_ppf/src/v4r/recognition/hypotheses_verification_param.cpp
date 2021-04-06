#include <v4r/recognition/hypotheses_verification_param.h>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
namespace po = boost::program_options;

namespace v4r {

void HV_Parameter::init(boost::program_options::options_description& desc, const std::string& section_name) {
  desc.add_options()((section_name + ".icp_iterations").c_str(),
                     po::value<int>(&icp_iterations_)->default_value(icp_iterations_),
                     "number of icp iterations. If 0, no pose refinement will be done");
  desc.add_options()((section_name + ".icp_max_correspondence").c_str(),
                     po::value<float>(&icp_max_correspondence_)->default_value(icp_max_correspondence_), "");
  desc.add_options()((section_name + ".icp_use_point_to_plane").c_str(),
                     po::value<bool>(&icp_use_point_to_plane_)->default_value(icp_use_point_to_plane_),
                     "if true, uses point-to-plane based ICP refinement. Otherwise point-to-point.");
  desc.add_options()(
      (section_name + ".icp_use_generalized_point_to_plane").c_str(),
      po::value<bool>(&icp_use_generalized_point_to_plane_)->default_value(icp_use_generalized_point_to_plane_),
      "if true, switches from default point-to-plane algorithm to Generalized ICP");
  desc.add_options()(
      (section_name + ".recompute_visible_points_after_icp").c_str(),
      po::value<bool>(&recompute_visible_points_after_icp_)->default_value(recompute_visible_points_after_icp_),
      "if true, recomputes visible points after pose refinement");
  desc.add_options()(
      (section_name + ".min_Euclidean_dist_between_centroids").c_str(),
      po::value<float>(&min_Euclidean_dist_between_centroids_)->default_value(min_Euclidean_dist_between_centroids_),
      "minimum Euclidean distances in meter between the centroids of two hypotheses of the same object model to be "
      "treated "
      "separately");
  desc.add_options()(
      (section_name + ".min_angular_degree_dist_between_hypotheses").c_str(),
      po::value<float>(&min_angular_degree_dist_between_hypotheses_)
          ->default_value(min_angular_degree_dist_between_hypotheses_),
      "minimum angular distance in degree between two hypotheses of the same object model to be treated separately");
  desc.add_options()(
      (section_name + ".clutter_regularizer").c_str(),
      po::value<float>(&clutter_regularizer_)
          ->default_value(clutter_regularizer_, boost::str(boost::format("%.2e") % clutter_regularizer_)),
      "The penalty multiplier used to penalize unexplained scene points within the clutter influence radius "
      "<i>radius_neighborhood_clutter_</i> of an explained scene point when they belong to the same smooth "
      "segment.");
  desc.add_options()(
      (section_name + ".inlier_threshold").c_str(),
      po::value<float>(&inlier_threshold_xyz_)
          ->default_value(inlier_threshold_xyz_, boost::str(boost::format("%.2e") % inlier_threshold_xyz_)),
      "Represents the maximum distance between model and scene points in order to state that a scene point is "
      "explained by a model point. Valid model points that do not have any corresponding scene point within this "
      "threshold are considered model outliers");
  desc.add_options()((section_name + ".inlier_threshold_normals_dotp").c_str(),
                     po::value<float>(&inlier_threshold_normals_dotp_)
                         ->default_value(inlier_threshold_normals_dotp_,
                                         boost::str(boost::format("%.2e") % inlier_threshold_normals_dotp_)),
                     "");
  desc.add_options()(
      (section_name + ".inlier_threshold_color").c_str(),
      po::value<float>(&inlier_threshold_color_)
          ->default_value(inlier_threshold_color_, boost::str(boost::format("%.2e") % inlier_threshold_color_)),
      "allowed chrominance (AB channel of LAB color space) variance for a point of an object hypotheses to be "
      "considered explained by a corresponding scene point (between 0 and 1, the higher the fewer objects get "
      "rejected)");
  desc.add_options()((section_name + ".histogram_specification").c_str(),
                     po::value<bool>(&use_histogram_specification_)->default_value(use_histogram_specification_), " ");
  desc.add_options()((section_name + ".ignore_color").c_str(),
                     po::value<bool>(&ignore_color_even_if_exists_)->default_value(ignore_color_even_if_exists_), " ");
  desc.add_options()(
      (section_name + ".color_comparison_method").c_str(),
      po::value<ColorComparisonMethod>(&color_comparison_method_)->default_value(color_comparison_method_),
      "method used for color comparison (0... CIE76, 1... CIE94, 2... CIEDE2000)");
  desc.add_options()(
      (section_name + ".occlusion_threshold").c_str(),
      po::value<float>(&occlusion_thres_)
          ->default_value(occlusion_thres_, boost::str(boost::format("%.2e") % occlusion_thres_)),
      "Threshold for a point to be considered occluded when model points are back-projected to the scene ( "
      "depends e.g. on sensor noise)");
  desc.add_options()((section_name + ".octree_resolution_m").c_str(),
                     po::value<float>(&octree_resolution_m_)->default_value(octree_resolution_m_),
                     "The resolution of the octree for computing visible ratio of objects (in meter)");
  desc.add_options()(
      (section_name + ".min_visible_ratio").c_str(),
      po::value<float>(&min_visible_ratio_)
          ->default_value(min_visible_ratio_, boost::str(boost::format("%.2e") % min_visible_ratio_)),
      "defines how much of the object has to be visible in order to be included in the verification stage");
  desc.add_options()(
      (section_name + ".min_ratio_smooth_cluster_explained").c_str(),
      po::value<float>(&min_ratio_cluster_explained_)
          ->default_value(min_ratio_cluster_explained_,
                          boost::str(boost::format("%.2e") % min_ratio_cluster_explained_)),
      " defines the minimum ratio a smooth cluster has to be explained by the visible points (given there are at "
      "least 100 points)");
  desc.add_options()((section_name + ".eps_angle_threshold").c_str(),
                     po::value<float>(&eps_angle_threshold_deg_)->default_value(eps_angle_threshold_deg_),
                     "smooth clustering parameter for the angle threshold");
  desc.add_options()((section_name + ".cluster_tolerance").c_str(),
                     po::value<float>(&cluster_tolerance_)->default_value(cluster_tolerance_),
                     "smooth clustering parameter for cluster_tolerance");
  desc.add_options()((section_name + ".curvature_threshold").c_str(),
                     po::value<float>(&curvature_threshold_)->default_value(curvature_threshold_),
                     "smooth clustering parameter for curvate");
  desc.add_options()(
      (section_name + ".check_smooth_clusters").c_str(),
      po::value<bool>(&check_smooth_clusters_)->default_value(check_smooth_clusters_),
      "if true, checks for each hypotheses how well it explains occupied smooth surface patches. Hypotheses are "
      "rejected if they only partially explain smooth clusters.");
  desc.add_options()((section_name + ".do_smoothing").c_str(),
                     po::value<bool>(&do_smoothing_)->default_value(do_smoothing_), "");
  desc.add_options()((section_name + ".do_erosion").c_str(), po::value<bool>(&do_erosion_)->default_value(do_erosion_),
                     "");
  desc.add_options()((section_name + ".erosion_radius").c_str(),
                     po::value<int>(&erosion_radius_)->default_value(erosion_radius_), "");
  desc.add_options()((section_name + ".smoothing_radius").c_str(),
                     po::value<int>(&smoothing_radius_)->default_value(smoothing_radius_), "");
  desc.add_options()((section_name + ".max_iterations").c_str(),
                     po::value<int>(&max_iterations_)->default_value(max_iterations_), "");
  desc.add_options()((section_name + ".min_points").c_str(),
                     po::value<size_t>(&min_points_)->default_value(min_points_), "");
  desc.add_options()((section_name + ".z_adaptive").c_str(), po::value<bool>(&z_adaptive_)->default_value(z_adaptive_),
                     "");
  desc.add_options()((section_name + ".min_pts_smooth_cluster_to_be_explained").c_str(),
                     po::value<size_t>(&min_pts_smooth_cluster_to_be_epxlained_)
                         ->default_value(min_pts_smooth_cluster_to_be_epxlained_),
                     "minimum number of points a cluster need to be explained by model "
                     "points to be considered for a check (avoids the fact that boundary "
                     " points of a smooth region can be close to an object");
  desc.add_options()((section_name + ".min_fitness").c_str(),
                     po::value<float>(&min_fitness_)->default_value(min_fitness_), "");
  desc.add_options()((section_name + ".min_dotproduct_model_normal_to_viewray").c_str(),
                     po::value<float>(&min_dotproduct_model_normal_to_viewray_)
                         ->default_value(min_dotproduct_model_normal_to_viewray_),
                     "points on the object models will be discarded from being marked visible if the point orientation "
                     "(surface normal) with respect to the viewray is smaller than this threshold. This reduces the "
                     "sensitivity of the visible object mask to small rotation changes (glancing intersection).");
  desc.add_options()((section_name + ".floor_z_min").c_str(),
                     po::value<float>(&floor_z_min_)->default_value(floor_z_min_),
                     "if reject_under_floor is set to true and if any point of a model is below this value then it is "
                     "under floor and rejected");
  desc.add_options()((section_name + ".reject_under_floor").c_str(),
                     po::value<bool>(&reject_under_floor_)->default_value(reject_under_floor_),
                     "if true, rejects models which have points under the floor level");

  scene_downsampler_param_.init(desc, section_name + ".scene_downsampling");

  desc.add_options()((section_name + ".outline_verification").c_str(),
                     po::value<bool>(&outline_verification_)->default_value(outline_verification_),
                     "if true, rejects models which depth outline does not match closely scene outlines");

  desc.add_options()((section_name + ".outline_verification_threshold").c_str(),
                     po::value<float>(&outline_verification_threshold_)->default_value(outline_verification_threshold_),
                     "threshold value for outline verification");

  depth_outlines_param_.init(desc, section_name + ".depth_outlines");
}
}  // namespace v4r