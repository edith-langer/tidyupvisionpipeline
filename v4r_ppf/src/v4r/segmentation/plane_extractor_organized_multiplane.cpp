#include <v4r/common/point_types.h>
#include <v4r/segmentation/plane_extractor_organized_multiplane.h>

#include <glog/logging.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/impl/instantiate.hpp>

namespace v4r {

template <typename PointT>
void OrganizedMultiPlaneExtractor<PointT>::do_compute(const boost::optional<const Eigen::Vector3f> &search_axis) {
  CHECK(cloud_ && cloud_->isOrganized()) << "Input cloud is not organized!";
  CHECK(normal_cloud_ && (normal_cloud_->points.size() == cloud_->points.size()));

  pcl::OrganizedMultiPlaneSegmentation<PointT, pcl::Normal, pcl::Label> mps;
  mps.setMinInliers(param_.min_num_plane_inliers_);
  mps.setAngularThreshold(param_.eps_angle_);
  mps.setDistanceThreshold(param_.distance_threshold_);
  mps.setInputNormals(normal_cloud_);
  mps.setInputCloud(cloud_);

  std::vector<pcl::PlanarRegion<PointT>, Eigen::aligned_allocator<pcl::PlanarRegion<PointT>>> regions;
  typename pcl::PlaneRefinementComparator<PointT, pcl::Normal, pcl::Label>::Ptr ref_comp(
      new pcl::PlaneRefinementComparator<PointT, pcl::Normal, pcl::Label>());
  ref_comp->setDistanceThreshold(param_.distance_threshold_, param_.z_adaptive_);
  ref_comp->setAngularThreshold(param_.eps_angle_);
  mps.setRefinementComparator(ref_comp);
  mps.segmentAndRefine(regions);

  all_planes_.clear();
  all_planes_.reserve(regions.size());
  for (const pcl::PlanarRegion<PointT> &reg : regions) {
    Eigen::Vector4f plane = reg.getCoefficients();

    // flip plane normal towards viewpoint
    Eigen::Vector3f z = Eigen::Vector3f::UnitZ();
    if (z.dot(plane.head(3)) > 0)
      plane = -plane;

    Eigen::Vector3f plane_normal = plane.head(3);
    plane_normal.normalize();

    if (search_axis) {
      float angle = acos(search_axis.get().dot(plane_normal));
      if (angle > param_.eps_angle_) {
        continue;
      }
    }

    all_planes_.push_back(plane);
  }
}

#define PCL_INSTANTIATE_OrganizedMultiPlaneExtractor(T) template class  OrganizedMultiPlaneExtractor<T>;
PCL_INSTANTIATE(OrganizedMultiPlaneExtractor, V4R_PCL_XYZ_POINT_TYPES)
}  // namespace v4r
