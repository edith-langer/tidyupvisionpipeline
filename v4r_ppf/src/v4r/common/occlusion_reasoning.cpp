#include <glog/logging.h>
#include <pcl/point_types.h>
#include <v4r/common/occlusion_reasoning.h>
#include <v4r/common/point_types.h>
#include <pcl/impl/instantiate.hpp>

namespace v4r {
template <typename PointTA, typename PointTB>
boost::dynamic_bitset<> computeVisiblePoints(const pcl::PointCloud<PointTA> &cloud_to_be_filtered,
                                             const pcl::PointCloud<PointTB> &occlusion_cloud,
                                             float occlusion_threshold) {
  CHECK(occlusion_cloud.isOrganized());
  CHECK(cloud_to_be_filtered.isOrganized());
  CHECK_EQ(occlusion_cloud.width, cloud_to_be_filtered.width);
  CHECK_EQ(occlusion_cloud.height, cloud_to_be_filtered.height);

  boost::dynamic_bitset<> px_is_visible_(occlusion_cloud.size(), false);

  const size_t n_points = cloud_to_be_filtered.size();
  for (size_t i = 0; i < n_points; ++i) {
    const auto &pt = cloud_to_be_filtered.points[i];
    if (!pcl::isFinite(pt))
      continue;
    const auto &pt_occ = occlusion_cloud.points[i];
    if (!pcl::isFinite(pt_occ) || pt.z < (pt_occ.z + occlusion_threshold)) {
      px_is_visible_.set(i);
    }
  }

  return px_is_visible_;
}

#define PCL_INSTANTIATE_OcclusionReasoner(TA, TB)                                                        \
  template  boost::dynamic_bitset<> computeVisiblePoints<TA, TB>(const pcl::PointCloud<TA> &, \
                                                                            const pcl::PointCloud<TB> &, float);
PCL_INSTANTIATE_PRODUCT(OcclusionReasoner, (V4R_PCL_XYZ_POINT_TYPES)(V4R_PCL_XYZ_POINT_TYPES))
}  // namespace v4r
