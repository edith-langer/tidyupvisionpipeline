/****************************************************************************
**
** Copyright (C) Aeolus Robotics, Inc.
**
****************************************************************************/

#pragma once

#include <Eigen/Dense>

namespace v4r {

/**
 * @brief compute rotation matrix that aligns two point pairs @f$ (a_s, a_t) @f$ and @f$ (b_s, b_t) @f$ such that (i)
 * the line connecting the two points @f$ (a_t - a_s) @f$ and @f$ (b_t - b_s) are aligned as well as the starting point
 * pair @f$ (a_s, a_t) @f$ coincides, and (ii) the surface normals of point pair @f$ a @f$ projected onto the plane
 * orthogonal to the aligned line aligns. The method is based on the paper "A Method for 6D Pose Estimation of Free-Form
 * Rigid Objects Using Point Pair Features on Range Data", Vidal et al, Sensors 2018 (Figure 4)
 * @tparam T floating point precision type
 * @param source_point_a source point a
 * @param source_normal_a source normal a
 * @param source_point_b source point b
 * @param target_point_a target point a
 * @param target_normal_a target normal a
 * @param target_point_b target point b
 * @return SE(3) rotation matrix that rotates source points into target points such that the pairs and their orientation
 * aligns
 */
template <typename T = float>
 Eigen::Matrix<T, 4, 4> alignOrientedPointPairs(const Eigen::Matrix<T, 4, 1> &source_point_a,
                                                           const Eigen::Matrix<T, 4, 1> &source_normal_a,
                                                           const Eigen::Matrix<T, 4, 1> &source_point_b,
                                                           const Eigen::Matrix<T, 4, 1> &target_point_a,
                                                           const Eigen::Matrix<T, 4, 1> &target_normal_a,
                                                           const Eigen::Matrix<T, 4, 1> &target_point_b);
}  // namespace v4r