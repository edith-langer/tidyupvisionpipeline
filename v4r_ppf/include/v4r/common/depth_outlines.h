#pragma once

#include <numeric>
#include <unordered_set>
#include <vector>

#include <pcl/features/organized_edge_detection.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/core/mat.hpp>

#include <v4r/common/depth_outlines_params.h>
// #include <v4r/common/pcl_opencv.h>

namespace v4r {

/**
 * A basic detection of outlines from organized point clouds.
 * It computes distance transform to allow efficient lookup for distance to edge
 */
class DepthOutlines {
 public:
  /**
   * Converts organized point cloud into set of outlines edges and computes distance transform
   * @tparam PointT type of point, need to have at least XYZ
   * @param scene organized point cloud
   * @param params can be used to change behaviour of edge detector
   */
  template <typename PointT>
  explicit DepthOutlines(const pcl::PointCloud<PointT>& scene,
                         DepthOutlinesParameter params = DepthOutlinesParameter()) {
    CHECK(scene.isOrganized());

    using OED = pcl::OrganizedEdgeFromRGBNormals<PointT, PointT, pcl::Label>;
    OED oed;
    oed.setDepthDisconThreshold(params.depth_discon_threshold_in_m);
    oed.setMaxSearchNeighbors(params.max_search_neighbors);

    int flags = OED::EDGELABEL_OCCLUDING;
    if (params.detect_occluded_edges)
      flags |= OED::EDGELABEL_OCCLUDED;
    if (params.detect_color_edges)
      flags |= OED::EDGELABEL_RGB_CANNY;
    if (params.detect_curvature_edges)
      flags |= OED::EDGELABEL_HIGH_CURVATURE;
    if (params.detect_nan_boundary)
      flags |= OED::EDGELABEL_NAN_BOUNDARY;

    oed.setEdgeType(flags);
    oed.setInputCloud({typename pcl::PointCloud<PointT>::ConstPtr(), &scene});
    if (params.detect_curvature_edges) {
      oed.setInputNormals({typename pcl::PointCloud<PointT>::ConstPtr(), &scene});
      oed.setHCCannyLowThreshold(params.curvature_low_threshold);
      oed.setHCCannyHighThreshold(params.curvature_high_threshold);
    }

    if (params.detect_color_edges) {
      oed.setRGBCannyLowThreshold(params.color_low_threshold);
      oed.setRGBCannyHighThreshold(params.color_high_threshold);
    }

    pcl::PointCloud<pcl::Label> labels;
    std::vector<pcl::PointIndices> edge_indices;
    oed.compute(labels, edge_indices);

    computeDistanceTransform(edge_indices, scene.width, scene.height);
  }

  /**
   * Query distance transform image and returns indices of all points that are in given L2 distance range
   * [min_distance, max_distance)
   * @param min_distance minimum value (included)
   * @param max_distance maximum value (excluded)
   * @return set of indices where distance to nearest outline is in [min_distance, max_distance)
   */
  std::vector<int> extractPointsNearOutline(unsigned min_distance = 1, unsigned max_distance = 8) const;

  /**
   * For a given set of indices returns squared distance to nearest outline
   * @param point_locations vector of uv points in distance transform image
   * @return squared distances for given indices
   */
  std::vector<float> extractDistancesSquared(const std::vector<int>& point_indices) const;

  /**
   * For a given set of uv locations returns squared distance to nearest outline
   * @param point_indices vector of indices in distance transform image
   * @return squared distances for given coordinates
   */
  std::vector<float> extractDistancesSquared(const std::vector<cv::Point2i>& point_locations) const;

  /**
   *
   * @return width of distance transform
   */
  int getWidth() const {
    return distance2border_.cols;
  }

  /**
   *
   * @return height of distance transform
   */
  int getHeight() const {
    return distance2border_.rows;
  }

 private:
  /**
   * Computes outlines map and distance transform based on given indices
   * @param edge_indices index of outline pixel
   * @param width of organized cloud
   * @param height of organized cloud
   */
  void computeDistanceTransform(const std::vector<pcl::PointIndices>& edge_indices, int width, int height);

  cv::Mat_<uchar> outlines_;
  cv::Mat distance2border_;
};
}  // namespace v4r