#pragma once
#include <pcl/point_cloud.h>

namespace v4r {

/**
 * @brief visualize clustering output
 * @param input cloud
 * @param indices of points belonging to the individual clusters
 */
template <typename PointT>
 void visualizeCluster(const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
                                  const std::vector<int> &cluster_indices,
                                  const std::string &window_title = "Segmentation results");

/**
 * @brief visualize clustering output
 * @param input cloud
 * @param indices of points belonging to the individual clusters
 * @param cluster_labels text labels for each cluster
 */
template <typename PointT>
 void visualizeClusters(const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
                                   const std::vector<std::vector<int>> &cluster_indices,
                                   const std::string &window_title = "Segmentation results",
                                   const std::vector<std::string> &cluster_labels = {});
}  // namespace v4r
