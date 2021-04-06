#include <pcl/filters/extract_indices.h>
#include <pcl/pcl_config.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_1_8/keypoints/uniform_sampling.h>
#include <v4r/common/point_types.h>
#include <v4r/segmentation/segmenter_euclidean.h>
#include <pcl/impl/instantiate.hpp>

namespace v4r {

template <typename PointT>
void EuclideanSegmenter<PointT>::segment() {
  pcl::EuclideanClusterExtraction<PointT> ec;
  std::vector<pcl::PointIndices> clusters_pcl;
  ec.setClusterTolerance(param_.distance_threshold_);
  ec.setMinClusterSize(param_.min_cluster_size_);
  ec.setMaxClusterSize(param_.max_cluster_size_);
  ec.setInputCloud(scene_);

  // no need to distinguish between pcl versions here, because nans are removed before
  // TODO: make this more generic, add this optionally to the segmenter class
  if (param_.downsample_before_segmentation_) {
    pcl_1_8::UniformSampling<PointT> us;
    us.setRadiusSearch(param_.downsample_before_segmentation_resolution_);
    us.setInputCloud(scene_);
    pcl::PointCloud<int> sampled_indices;
    us.compute(sampled_indices);

    std::vector<int> scene_sampled_indices;
    scene_sampled_indices.reserve(sampled_indices.size());
    for (size_t i = 0; i < sampled_indices.size(); i++) {
      const int idx = sampled_indices.points[i];
      if (pcl::isFinite(scene_->points[idx])) {
        scene_sampled_indices.push_back(idx);
      }
    }
    scene_sampled_indices.shrink_to_fit();

    pcl::PointIndices::Ptr indices(new pcl::PointIndices());
    indices->header = scene_->header;
    indices->indices = scene_sampled_indices;
    ec.setIndices(indices);
  }
#if PCL_VERSION_COMPARE(<, 1, 9, 0)
  else {
    pcl::PointIndices::Ptr indices(new pcl::PointIndices());
    indices->header = scene_->header;
    pcl::removeNaNFromPointCloud(*scene_, indices->indices);
    ec.setIndices(indices);
  }
#endif
  ec.extract(clusters_pcl);
  clusters_.resize(clusters_pcl.size());
  for (size_t i = 0; i < clusters_pcl.size(); i++)
    clusters_[i] = clusters_pcl[i].indices;
}

#define PCL_INSTANTIATE_EuclideanSegmenter(T) template class  EuclideanSegmenter<T>;
PCL_INSTANTIATE(EuclideanSegmenter, V4R_PCL_XYZ_POINT_TYPES)
}  // namespace v4r