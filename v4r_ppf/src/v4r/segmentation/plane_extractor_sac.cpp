#include <v4r/common/point_types.h>
#include <v4r/segmentation/plane_extractor_sac.h>

#include <glog/logging.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/organized.h>
#include <pcl/impl/instantiate.hpp>

namespace v4r {

template <typename PointT>
void SACPlaneExtractor<PointT>::do_compute(const boost::optional<const Eigen::Vector3f> &search_axis) {
  CHECK(cloud_) << "Input cloud is not set!";

  all_planes_.clear();

  typename std::shared_ptr<pcl::SACSegmentation<PointT>> sac;
  if (getRequiresNormals()) {
    std::shared_ptr<pcl::SACSegmentationFromNormals<PointT, pcl::Normal>> sac_n(
        new pcl::SACSegmentationFromNormals<PointT, pcl::Normal>);
    sac_n->setInputNormals(normal_cloud_);
    sac_n->setNormalDistanceWeight(0.1);
    sac = sac_n;
  } else
    sac.reset(new pcl::SACSegmentation<PointT>);

  sac->setOptimizeCoefficients(param_.optimize_cofficients_);
  sac->setModelType(param_.model_type_);
  sac->setMethodType(param_.method_type_);
  sac->setMaxIterations(param_.max_iterations_);
  sac->setDistanceThreshold(param_.distance_threshold_);
  sac->setProbability(param_.probability_);
  sac->setEpsAngle(param_.eps_angle_);

  if (search_axis)
    sac->setAxis(search_axis.get());

  typename pcl::search::Search<PointT>::Ptr searcher;
  if (cloud_->isOrganized())
    searcher.reset(new pcl::search::OrganizedNeighbor<PointT>());
  else
    searcher.reset(new pcl::search::KdTree<PointT>());

  typename pcl::PointCloud<PointT>::Ptr filtered_cloud(new pcl::PointCloud<PointT>(*cloud_));

  do {
    // Segment the largest planar component from the remaining cloud
    sac->setInputCloud(filtered_cloud);
    searcher->setInputCloud(filtered_cloud);
    sac->setSamplesMaxDist(param_.samples_max_distance_, searcher);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    sac->segment(*inliers, *coefficients);

    if (inliers->indices.size() < param_.min_num_plane_inliers_)
      break;

    all_planes_.push_back(Eigen::Vector4f(coefficients->values[0], coefficients->values[1], coefficients->values[2],
                                          coefficients->values[3]));
    plane_inliers_.push_back(inliers->indices);

    for (int idx : inliers->indices) {
      PointT &p = filtered_cloud->points[idx];
      p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
    }
  } while (param_.compute_all_planes_);
}

#define PCL_INSTANTIATE_SACPlaneExtractor(T) template class  SACPlaneExtractor<T>;
PCL_INSTANTIATE(SACPlaneExtractor, V4R_PCL_XYZ_POINT_TYPES)
}  // namespace v4r
