#include <glog/logging.h>
#include <PPFRecognizerParameter.h>

namespace v4r {
namespace apps {

void PPFRecognizerParameter::init(boost::program_options::options_description &desc,
                                     const std::string &section_name) {
  desc.add_options()("models", po::value<std::vector<std::string>>(&object_models_)->multitoken(),
                     "Names of object models to load from the models directory. If empty or not set, all object models "
                     "will be loaded from the object models directory.");
  desc.add_options()("camera_calibration_file",
                     po::value<bf::path>(&camera_calibration_file_)->default_value(camera_calibration_file_),
                     "Calibration file of RGB Camera containing intrinsic parameters");
  desc.add_options()(
      "rgb_depth_overlap_image",
      po::value<bf::path>(&rgb_depth_overlap_image_)->default_value(rgb_depth_overlap_image_),
      "binary image mask depicting the overlap of the registered RGB and depth image. Pixel values of 255 (white) "
      "indicate that the pixel is visible by both RGB and depth after registration. Black pixel are only seen by the "
      "RGB image due to the physical displacement between RGB and depth sensor.");
  desc.add_options()((section_name + ".do_ppf").c_str(), po::value<bool>(&do_ppf_)->default_value(do_ppf_),
                     "Enables PPF pipeline");
  desc.add_options()((section_name + ".remove_planes").c_str(),
                     po::value<bool>(&remove_planes_)->default_value(remove_planes_),
                     "if enabled, removes the dominant plane in the input cloud (given there are at least N inliers)");
  desc.add_options()((section_name + ".chop_z").c_str(), po::value<double>(&chop_z_)->default_value(chop_z_),
                     "Cut-off distance in z-direction of the camera");
  desc.add_options()(
      (section_name + ".min_height_above_ground").c_str(),
      po::value<float>(&min_height_above_ground_)->default_value(min_height_above_ground_),
      "minimum height above ground for input points to be considered (only used if transform_to_world is set)");
  desc.add_options()(
      (section_name + ".max_height_above_ground").c_str(),
      po::value<float>(&max_height_above_ground_)->default_value(max_height_above_ground_),
      "maximum height above ground for input points to be considered (only used if transform_to_world is set)");
  desc.add_options()((section_name + ".use_multiview").c_str(),
                     po::value<bool>(&use_multiview_)->default_value(use_multiview_), "");
  desc.add_options()((section_name + ".use_multiview_hv").c_str(),
                     po::value<bool>(&use_multiview_hv_)->default_value(use_multiview_hv_), "");
  desc.add_options()((section_name + ".use_multiview_with_kp_correspondence_transfer").c_str(),
                     po::value<bool>(&use_multiview_with_kp_correspondence_transfer_)
                         ->default_value(use_multiview_with_kp_correspondence_transfer_),
                     "if true, transfers keypoints instead of full hypotheses (see Faeulhammer et al, ICRA 2015)");
  desc.add_options()((section_name + ".multiview_max_views").c_str(),
                     po::value<size_t>(&multiview_max_views_)->default_value(multiview_max_views_),
                     "Maximum number of views used for multi-view recognition (if more views are available, "
                     "information from oldest views will be ignored)");
  desc.add_options()((section_name + ".use_change_detection").c_str(),
                     po::value<bool>(&use_change_detection_)->default_value(use_change_detection_), "");
  desc.add_options()(
      (section_name + ".remove_non_upright_objects").c_str(),
      po::value<bool>(&remove_non_upright_objects_)->default_value(remove_non_upright_objects_),
      "remove all hypotheses that are not standing upright on a support plane (support plane extraction must be "
      "enabled)");
  desc.add_options()(
      (section_name + ".max_model_diameter_to_min_plane_ratio").c_str(),
      po::value(&max_model_diameter_to_min_plane_ratio_)->default_value(max_model_diameter_to_min_plane_ratio_),
      "multiplier for max model diameter which is used for plane removal");

  desc.add_options()("icp_iterations",
                     po::value<size_t>(&icp_iterations_)->default_value(icp_iterations_),
                     "ICP iterations. Only used if hypotheses are not verified. Otherwise ICP is done inside HV");
  desc.add_options()("icp_max_corr_dist",
                     po::value<double>(&icp_max_corr_dist_)->default_value(icp_max_corr_dist_),
                     "ICP parameter: max allowed distance between corresponding points. Only used if hypotheses are not verified. Otherwise ICP is done inside HV");
  desc.add_options()("icp_transf_eps",
                     po::value<double>(&icp_transf_eps_)->default_value(icp_transf_eps_),
                     "ICP parameter: terminate if difference between the previous transformation and the current estimated transformation is smaller. Only used if hypotheses are not verified. Otherwise ICP is done inside HV");
  desc.add_options()("skip_verification", po::bool_switch(&skip_verification_),
                     "if true, skips verification (only hypotheses generation)");
//  desc.add_options()("min_graph_conf_thr",
//                     po::value<double>(&min_graph_conf_thr_)->default_value(min_graph_conf_thr_),
//                     "If the confidence is below that value, the hypothesis gets rejected");
//  desc.add_options()("min_result_conf_thr",
//                     po::value<double>(&min_result_conf_thr_)->default_value(min_result_conf_thr_),
//                     "If the confidence is below that value, the hypothesis gets rejected");
  desc.add_options()("min_fitness_weight_thr",
                     po::value<double>(&min_fitness_weight_thr_)->default_value(min_fitness_weight_thr_),
                     "If the confidence is below that value, the hypothesis gets rejected");
  desc.add_options()("avg_fitness_weight_thr",
                     po::value<double>(&avg_fitness_weight_thr_)->default_value(avg_fitness_weight_thr_),
                     "If the confidence is below that value, the hypothesis gets rejected");

  normal_estimator_param_.init(desc, section_name + ".normal_estimator");
  plane_filter_.init(desc, "plane_filter");
  ppf_rec_pipeline_.init(desc, "ppf_pipeline");
  hv_.init(desc, "hv");
//   multiview_.init(desc, "mv");
}
}  // namespace apps
}  // namespace v4r
