#include <v4r/geometry/geometry.h>
#include <Eigen/Geometry>

namespace v4r {

template <typename T>
Eigen::Matrix<T, 4, 4> alignOrientedPointPairs(const Eigen::Matrix<T, 4, 1> &source_point_a,
                                               const Eigen::Matrix<T, 4, 1> &source_normal_a,
                                               const Eigen::Matrix<T, 4, 1> &source_point_b,
                                               const Eigen::Matrix<T, 4, 1> &target_point_a,
                                               const Eigen::Matrix<T, 4, 1> &target_normal_a,
                                               const Eigen::Matrix<T, 4, 1> &target_point_b) {
  using Matrix4 = Eigen::Matrix<T, 4, 4>;
  using Vector4 = Eigen::Matrix<T, 4, 1>;

  const Vector4 d_target_normalized = (target_point_a - target_point_b).normalized();
  const Vector4 d_source = source_point_a - source_point_b;

  // first align lines connecting a and b
  Matrix4 tf = Matrix4::Identity();
  tf.template block<3, 3>(0, 0) =
      Eigen::Quaternion<T>::FromTwoVectors(d_source.template head<3>(), d_target_normalized.template head<3>())
          .toRotationMatrix();

  const Vector4 source_normal_a_aligned = tf * source_normal_a;

  // next project surface normals of points k onto the plane perpendicular to d_target ( = d_source), and then
  // rotate these projected surface normals such that they align
  const Vector4 target_normal_a_projected =
      (target_normal_a - target_normal_a.dot(d_target_normalized) * d_target_normalized);

  const Vector4 source_normal_a_projected =
      (source_normal_a_aligned - source_normal_a_aligned.dot(d_target_normalized) * d_target_normalized);

  Matrix4 tf2 = Matrix4::Identity();
  tf2.template block<3, 3>(0, 0) = Eigen::Quaternion<T>::FromTwoVectors(source_normal_a_projected.template head<3>(),
                                                                        target_normal_a_projected.template head<3>())
                                       .toRotationMatrix();

  Matrix4 tf_trans_source = Matrix4::Identity();
  tf_trans_source.template block<4, 1>(0, 3) = -source_point_b;

  Matrix4 tf_trans_target = Matrix4::Identity();
  tf_trans_target.template block<4, 1>(0, 3) = -target_point_b;

  return tf_trans_target * tf2 * tf * tf_trans_source;
}

template  Eigen::Matrix<float, 4, 4> alignOrientedPointPairs<float>(
    const Eigen::Matrix<float, 4, 1> &, const Eigen::Matrix<float, 4, 1> &, const Eigen::Matrix<float, 4, 1> &,
    const Eigen::Matrix<float, 4, 1> &, const Eigen::Matrix<float, 4, 1> &, const Eigen::Matrix<float, 4, 1> &);

}  // namespace v4r