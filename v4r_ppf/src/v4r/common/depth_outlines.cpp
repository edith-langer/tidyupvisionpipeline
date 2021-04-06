#include <v4r/common/depth_outlines.h>
#include <opencv2/imgproc.hpp>

namespace v4r {

void DepthOutlinesParameter::init(boost::program_options::options_description& desc, const std::string& section_name) {
  namespace po = boost::program_options;
  auto option = [&](std::string field_name, auto& value, const char* help_msg) {
    desc.add_options()((section_name + '.' + field_name).c_str(), po::value<>(&value)->default_value(value), help_msg);
  };

  option("depth_discon_threshold_in_m", depth_discon_threshold_in_m, "depth discontinuity threshold in meters");
  option("max_search_neighbors", max_search_neighbors,
         "max n of neighbors for deciding between occluder/occluded edges");
  option("detect_occluded_edges", detect_occluded_edges, "if true, detect occluded edges");
  option("detect_color_edges", detect_color_edges, "if true, detect color edges");
  option("detect_curvature_edges", detect_occluded_edges, "if true, detect curvature edges");
  option("detect_nan_boundary", detect_nan_boundary, "if true, detect NaN boundary");
  option("color_low_threshold", color_low_threshold, "low threshold for Canny RGB edge detector");
  option("color_high_threshold", color_high_threshold, "high threshold for Canny RGB edge detector");
  option("curvature_low_threshold", curvature_low_threshold, "low threshold for Canny high curvature detector");
  option("curvature_high_threshold", curvature_high_threshold, "high threshold for Canny high curvature detector");
}

void DepthOutlines::computeDistanceTransform(const std::vector<pcl::PointIndices>& edge_indices, int width,
                                             int height) {
  outlines_ = cv::Mat_<uchar>(height, width);
  outlines_.setTo(255);

  for (size_t j = 0; j < edge_indices.size(); j++) {
    for (int edge_px_id : edge_indices[j].indices) {
      outlines_.at<uchar>(edge_px_id) = 0;
    }
  }

  cv::distanceTransform(outlines_, distance2border_, CV_DIST_L2, cv::DIST_MASK_PRECISE);
}

std::vector<int> DepthOutlines::extractPointsNearOutline(unsigned min_distance, unsigned max_distance) const {
  const float min_val = min_distance;
  const float max_val = max_distance;
  std::vector<int> locations;
  const int width = distance2border_.cols;
  for (int v = 0; v < distance2border_.rows; ++v) {
    for (int u = 0; u < width; ++u) {
      float value = distance2border_.at<float>(v, u);
      if (value >= min_val && value < max_val) {
        locations.emplace_back(u + v * width);
      }
    }
  }
  return locations;
}

std::vector<float> DepthOutlines::extractDistancesSquared(const std::vector<int>& point_indices) const {
  std::vector<float> result;
  result.reserve(point_indices.size());
  for (auto i : point_indices) {
    float distance = distance2border_.at<float>(i);
    result.emplace_back(distance * distance);
  }
  return result;
}

std::vector<float> DepthOutlines::extractDistancesSquared(const std::vector<cv::Point2i>& point_locations) const {
  std::vector<float> result;
  result.reserve(point_locations.size());
  for (auto i : point_locations) {
    float distance = distance2border_.at<float>(i);
    result.emplace_back(distance * distance);  // todo: probably should also consider how far away point is
  }
  return result;
}

}  // namespace v4r