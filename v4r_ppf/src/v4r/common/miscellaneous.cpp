#include <v4r/common/miscellaneous.h>
#include <v4r/common/point_types.h>

#include <glog/logging.h>
#include <pcl/common/centroid.h>
#include <pcl/common/eigen.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/impl/instantiate.hpp>

namespace v4r {

inline float rad2deg(float alpha) {
  return (alpha * 57.29578f);
}

bool incrementVector(const std::vector<bool> &v, std::vector<bool> &inc_v) {
  inc_v = v;

  bool overflow = std::all_of(v.begin(), v.end(), [](const auto b) { return b; });

  bool carry = v.back();
  inc_v.back() = !v.back();
  for (int bit = v.size() - 2; bit >= 0; bit--) {
    inc_v[bit] = v[bit] != carry;
    carry = v[bit] && carry;
  }
  return overflow;
}

boost::dynamic_bitset<> computeMaskFromIndexMap(const Eigen::MatrixXi &image_map, size_t nr_points) {
  boost::dynamic_bitset<> mask(nr_points, 0);
  for (int i = 0; i < image_map.size(); i++) {
    int val = *(image_map.data() + i);
    if (val >= 0)
      mask.set(val);
  }

  return mask;
}

Eigen::Matrix3f computeRotationMatrixToAlignVectors(const Eigen::Vector3f &src, const Eigen::Vector3f &target) {
  Eigen::Vector3f A = src;
  Eigen::Vector3f B = target;

  A.normalize();
  B.normalize();

  float c = A.dot(B);

  if (c > 1.f - std::numeric_limits<float>::epsilon())
    return Eigen::Matrix3f::Identity();

  if (c < -1.f + std::numeric_limits<float>::epsilon()) {
    LOG(ERROR) << "Computing a rotation matrix of two opposite vectors is not supported by this equation. The returned "
                  "rotation matrix won't be a proper rotation matrix!";
    return -Eigen::Matrix3f::Identity();  // flip
  }

  const Eigen::Vector3f v = A.cross(B);

  Eigen::Matrix3f vx;
  vx << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;

  return Eigen::Matrix3f::Identity() + vx + vx * vx / (1.f + c);
}

template <typename PointT>
 void computePointCloudProperties(const pcl::PointCloud<PointT> &cloud, Eigen::Vector4f &centroid,
                                             Eigen::Vector3f &elongationsXYZ, Eigen::Matrix4f &covariancePose,
                                             const std::vector<int> &indices) {
  EIGEN_ALIGN16 Eigen::Matrix3f covariance_matrix;
  EIGEN_ALIGN16 Eigen::Vector3f eigenValues;
  EIGEN_ALIGN16 Eigen::Matrix3f eigenVectors;
  EIGEN_ALIGN16 Eigen::Matrix3f eigenBasis;

  if (indices.empty())
    computeMeanAndCovarianceMatrix(cloud, covariance_matrix, centroid);
  else
    computeMeanAndCovarianceMatrix(cloud, indices, covariance_matrix, centroid);

  pcl::eigen33(covariance_matrix, eigenVectors, eigenValues);

  // create orthonormal rotation matrix from eigenvectors
  eigenBasis.col(0) = eigenVectors.col(0).normalized();
  float dotp12 = eigenVectors.col(1).dot(eigenBasis.col(0));
  Eigen::Vector3f eig2 = eigenVectors.col(1) - dotp12 * eigenBasis.col(0);
  eigenBasis.col(1) = eig2.normalized();
  Eigen::Vector3f eig3 = eigenBasis.col(0).cross(eigenBasis.col(1));
  eigenBasis.col(2) = eig3.normalized();

  // transform cluster into origin and align with eigenvectors
  Eigen::Matrix4f tf_rot = Eigen::Matrix4f::Identity();
  tf_rot.block<3, 3>(0, 0) = eigenBasis.transpose();
  Eigen::Matrix4f tf_trans = Eigen::Matrix4f::Identity();
  tf_trans.block<3, 1>(0, 3) = -centroid.topRows(3);

  covariancePose = tf_rot * tf_trans;
  //    Eigen::Matrix4f tf_rot = tf_rot_inv.inverse();

  // compute max elongations
  pcl::PointCloud<PointT> eigenvec_aligned;

  if (indices.empty())
    pcl::copyPointCloud(cloud, eigenvec_aligned);
  else
    pcl::copyPointCloud(cloud, indices, eigenvec_aligned);

  pcl::transformPointCloud(eigenvec_aligned, eigenvec_aligned, tf_rot * tf_trans);

  float xmin, ymin, xmax, ymax, zmin, zmax;
  xmin = ymin = xmax = ymax = zmin = zmax = 0.f;
  for (size_t pt = 0; pt < eigenvec_aligned.points.size(); pt++) {
    const PointT &p = eigenvec_aligned.points[pt];
    if (p.x < xmin)
      xmin = p.x;
    if (p.x > xmax)
      xmax = p.x;
    if (p.y < ymin)
      ymin = p.y;
    if (p.y > ymax)
      ymax = p.y;
    if (p.z < zmin)
      zmin = p.z;
    if (p.z > zmax)
      zmax = p.z;
  }

  elongationsXYZ(0) = xmax - xmin;
  elongationsXYZ(1) = ymax - ymin;
  elongationsXYZ(2) = zmax - zmin;
}

bool similarPoseExists(const Eigen::Matrix4f &pose,
                       const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &existing_poses,
                       float required_viewpoint_change_degree) {
  bool similar_pose_exists = false;

  for (const Eigen::Matrix4f &ep : existing_poses) {
    Eigen::Vector3f v1 = pose.block<3, 1>(0, 0);
    Eigen::Vector3f v2 = ep.block<3, 1>(0, 0);
    v1.normalize();
    v2.normalize();
    float dotp = v1.dot(v2);
    const Eigen::Vector3f crossp = v1.cross(v2);

    float rel_angle_deg = rad2deg(acos(dotp));
    if (crossp(2) < 0.f)
      rel_angle_deg = 360.f - rel_angle_deg;

    if (rel_angle_deg < required_viewpoint_change_degree) {
      similar_pose_exists = true;
      break;
    }
  }
  return similar_pose_exists;
}

template <typename PointT>
float computePointcloudDiameter(const pcl::PointCloud<PointT> &cloud) {
  auto mp = cloud.getMatrixXfMap(3, sizeof(PointT) / sizeof(float), 0);
  try {
    return std::sqrt((((mp.transpose() * mp * -2).colwise() + mp.colwise().squaredNorm().transpose()).rowwise() +
                      mp.colwise().squaredNorm())
                         .maxCoeff());
  } catch (std::bad_alloc &) {
    LOG(FATAL) << "Point cloud size is too large; impossible to allocate memory for the matrix of pairwise distances";
  }
}

#define PCL_INSTANTIATE_computePointCloudProperties(T)                                                     \
  template  void computePointCloudProperties<T>(const pcl::PointCloud<T> &, Eigen::Vector4f &,  \
                                                           Eigen::Vector3f &, Eigen::Matrix4f &eigenBasis, \
                                                           const std::vector<int> &);
PCL_INSTANTIATE(computePointCloudProperties, V4R_PCL_XYZ_POINT_TYPES)

template  std::vector<size_t> createIndicesFromMask(const boost::dynamic_bitset<> &mask, bool invert);

template  std::vector<int> createIndicesFromMask(const boost::dynamic_bitset<> &mask, bool invert);

#define PCL_INSTANTIATE_computePointcloudDiameter(T) \
  template  float computePointcloudDiameter<T>(const pcl::PointCloud<T> &);
PCL_INSTANTIATE(computePointcloudDiameter, V4R_PCL_XYZ_POINT_TYPES)
}  // namespace v4r
