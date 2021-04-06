#include <v4r/common/noise_models.h>

#include <glog/logging.h>
#include <omp.h>

namespace v4r {

template <typename PointT>
void NguyenNoiseModel<PointT>::computeNoiseLevel(const PointT &pt, const pcl::Normal &n, float &sigma_lateral,
                                                 float &sigma_axial, float focal_length) {
  if (!pcl::isFinite(pt) || !pcl::isFinite(n)) {
    sigma_lateral = sigma_axial = std::numeric_limits<float>::max();
  }

  // origin to point
  // Eigen::Vector3f o2p = input_->points[i].getVector3fMap() * -1.f;
  const Eigen::Vector4f o2p = Eigen::Vector4f::UnitZ() * -1.f;
  float angle = pcl::rad2deg(acos(o2p.dot(n.getNormalVector4fMap())));

  if (angle > 85.f)
    angle = 85.f;

  float sigma_lateral_px = (0.8f + 0.034f * angle / (90.f - angle)) * pt.z / focal_length;  // in pixel
  sigma_lateral = sigma_lateral_px * pt.z * 1.f;                                            // in meters
  sigma_axial = 0.0012f + 0.0019f * (pt.z - 0.4f) * (pt.z - 0.4f) +
                0.0001f * angle * angle / (sqrt(pt.z) * (90.f - angle) * (90.f - angle));  // in meters
}

template <typename PointT>
std::vector<std::vector<float>> NguyenNoiseModel<PointT>::compute(
    const typename pcl::PointCloud<PointT>::ConstPtr &input, const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
    float focal_length) {
  CHECK(input->isOrganized());
  std::vector<std::vector<float>> pt_properties(input->size(), std::vector<float>(2));

#pragma omp parallel for schedule(dynamic)
  for (size_t i = 0; i < input->size(); i++) {
    computeNoiseLevel(input->points[i], normals->points[i], pt_properties[i][0], pt_properties[i][1], focal_length);
  }
  return pt_properties;
}

template class NguyenNoiseModel<pcl::PointXYZRGB>;
// template class  NguyenNoiseModel<pcl::PointXYZ>;
}  // namespace v4r
