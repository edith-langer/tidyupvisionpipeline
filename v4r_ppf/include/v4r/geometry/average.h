/****************************************************************************
**
** Copyright (C) 2019 TU Wien, ACIN, Vision 4 Robotics (V4R) group
** Contact: v4r.acin.tuwien.ac.at
**
** This file is part of V4R
**
** V4R is distributed under dual licenses - GPLv3 or closed source.
**
** GNU General Public License Usage
** V4R is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published
** by the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
**
** V4R is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
** GNU General Public License for more details.
**
** Please review the following information to ensure the GNU General Public
** License requirements will be met: https://www.gnu.org/licenses/gpl-3.0.html.
**
**
** Commercial License Usage
** If GPL is not suitable for your project, you must purchase a commercial
** license to use V4R. Licensees holding valid commercial V4R licenses may
** use this file in accordance with the commercial license agreement
** provided with the Software or, alternatively, in accordance with the
** terms contained in a written agreement between you and TU Wien, ACIN, V4R.
** For licensing terms and conditions please contact office<at>acin.tuwien.ac.at.
**
**
** The copyright holder additionally grants the author(s) of the file the right
** to use, copy, modify, merge, publish, distribute, sublicense, and/or
** sell copies of their contributions without any restrictions.
**
****************************************************************************/

#pragma once

#include <vector>

#include <glog/logging.h>

#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

namespace v4r {

namespace geometry {

/// Compute quaternion average.
template <typename DataType>
class QuaternionAverage {
 public:
  using Matrix = Eigen::Matrix<DataType, 4, 4>;
  using Quaternion = Eigen::Quaternion<DataType>;

  /// Add a single quaternion to the average.
  template <typename QuaternionType>
  void add(const QuaternionType& quaternion) {
    Eigen::Matrix<DataType, 1, 4> q(quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z());
    A_ += q.transpose() * q;
    count_++;
  }

  /// Add a range of quaternions to the average.
  template <typename ForwardIterator>
  void add(const ForwardIterator& begin, const ForwardIterator& end) {
    for (ForwardIterator it = begin; it != end; ++it)
      add(*it);
  }

  /// Get the average.
  Quaternion get() const {
    CHECK(count_) << "Can not get average because no quaternions were added";
    Eigen::EigenSolver<Matrix> es(A_ / count_);
    Eigen::Matrix<std::complex<DataType>, 4, 1> mat(es.eigenvalues());
    int index;
    mat.real().maxCoeff(&index);
    Eigen::Matrix<DataType, 4, 1> ev(es.eigenvectors().real().template block<4, 1>(0, index));
    return Quaternion(ev(0), ev(1), ev(2), ev(3));
  }

 private:
  Matrix A_ = Matrix::Zero();
  unsigned int count_ = 0;
};

/// Compute average orientation using quaternions in a container (that has begin/end).
template <typename DataType, typename Container>
typename QuaternionAverage<DataType>::Quaternion averageQuaternions(const Container& container) {
  QuaternionAverage<DataType> average;
  average.add(container.begin(), container.end());
  return average.get();
}

/// Compute position average.
template <typename DataType>
class PositionAverage {
 public:
  using Vector = Eigen::Matrix<DataType, 3, 1>;

  /// Add a single position to the average.
  template <typename PositionType>
  void add(const PositionType& position) {
    xyz_ += position.template cast<DataType>();
    count_++;
  }

  /// Add a range of positions to the average.
  template <typename ForwardIterator>
  void add(const ForwardIterator& begin, const ForwardIterator& end) {
    for (ForwardIterator it = begin; it != end; ++it)
      add(*it);
  }

  /// Get the average.
  Vector get() const {
    CHECK(count_) << "Can not get average because no positions were added";
    return xyz_ / count_;
  }

 private:
  Vector xyz_ = Vector::Zero();
  unsigned int count_ = 0;
};

/// Compute average position using vectors in a container (that has begin/end).
template <typename DataType, typename Container>
typename PositionAverage<DataType>::Vector averagePositions(const Container& container) {
  PositionAverage<DataType> average;
  average.add(container.begin(), container.end());
  return average.get();
}

/// Compute average of affine transforms.
template <typename DataType>
class TransformAverage {
 public:
  using Transform = Eigen::Transform<DataType, 3, Eigen::Affine>;

  /// Add a single transform to the average.
  template <typename TransformType>
  void add(const TransformType& transform) {
    position_average_.add(transform.translation());
    quaternion_average_.add(Eigen::Quaternion<DataType>(transform.rotation().template cast<DataType>()));
  }

  /// Add a range of transforms to the average.
  template <typename ForwardIterator>
  void add(const ForwardIterator& begin, const ForwardIterator& end) {
    for (auto it = begin; it != end; ++it) {
      position_average_.add(it->translation());
      quaternion_average_.add(Eigen::Quaternion<DataType>(it->rotation().template cast<DataType>()));
    }
  }

  /// Get the average.
  Transform get() const {
    return Eigen::Translation<DataType, 3>(position_average_.get()) * quaternion_average_.get();
  }

 private:
  PositionAverage<DataType> position_average_;
  QuaternionAverage<DataType> quaternion_average_;
};

/// Compute average transform using transforms in a container (that has begin/end).
template <typename DataType, typename Container>
typename TransformAverage<DataType>::Transform averageTransforms(const Container& container) {
  TransformAverage<DataType> average;
  average.add(container.begin(), container.end());
  return average.get();
}

}  // namespace geometry

}  // namespace v4r
