#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <v4r/common/Clustering.h>
#include <v4r/common/point_types.h>
#include <pcl/impl/instantiate.hpp>

namespace v4r {

template <typename PointT>
Cluster::Cluster(const pcl::PointCloud<PointT> &cloud, const std::vector<int> &indices, bool compute_properties)
: table_plane_set_(false), indices_(indices) {
  if (compute_properties) {
    Eigen::Matrix3f covariance_matrix;

    if (indices.empty())
      computeMeanAndCovarianceMatrix(cloud, covariance_matrix, centroid_);
    else
      computeMeanAndCovarianceMatrix(cloud, indices, covariance_matrix, centroid_);

    pcl::eigen33(covariance_matrix, eigen_vectors_, eigen_values_);

    // transform cluster into origin and align with eigenvectors
    Eigen::Matrix4f tf_rot = Eigen::Matrix4f::Identity();
    tf_rot.block<3, 3>(0, 0) = eigen_vectors_.transpose();
    Eigen::Matrix4f tf_trans = Eigen::Matrix4f::Identity();
    tf_trans.block<3, 1>(0, 3) = -centroid_.topRows(3);
    eigen_pose_alignment_ = tf_rot * tf_trans;

    // compute max elongations
    pcl::PointCloud<PointT> cloud_axis_aligned;

    if (indices.empty())
      pcl::transformPointCloud(cloud, cloud_axis_aligned, eigen_pose_alignment_);
    else {
      pcl::copyPointCloud(cloud, indices, cloud_axis_aligned);
      pcl::transformPointCloud(cloud_axis_aligned, cloud_axis_aligned, eigen_pose_alignment_);
    }

    Eigen::Vector4f min, max;
    pcl::getMinMax3D(cloud_axis_aligned, min, max);
    elongation_ = (max - min).head(3);
  }
}

#define PCL_INSTANTIATE_Cluster(T) \
  template Cluster::Cluster(const pcl::PointCloud<T> &, const std::vector<int> &, bool);
PCL_INSTANTIATE(Cluster, V4R_PCL_XYZ_POINT_TYPES)

}  // namespace v4r