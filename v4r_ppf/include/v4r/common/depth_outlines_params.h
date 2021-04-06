#pragma once

#include <boost/program_options.hpp>
#include <string>

namespace v4r {

/**
 * POD with settings for DepthOutlines class.
 */
struct DepthOutlinesParameter {
  float depth_discon_threshold_in_m = 0.02f;
  int max_search_neighbors = 100;
  bool detect_occluded_edges = false;
  bool detect_color_edges = false;
  bool detect_curvature_edges = false;
  bool detect_nan_boundary = false;

  int color_low_threshold = 40;
  int color_high_threshold = 100;

  float curvature_low_threshold = 0.5f;
  float curvature_high_threshold = 1.1f;

  void init(boost::program_options::options_description& desc, const std::string& section_name);
};

}  // namespace v4r
