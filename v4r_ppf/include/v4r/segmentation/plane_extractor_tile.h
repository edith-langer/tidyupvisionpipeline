/****************************************************************************
**
** Copyright (C) 2017 TU Wien, ACIN, Vision 4 Robotics (V4R) group
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

/**
 * @file plane_extractor_tile.h
 * @author Simon Schreiberhuber (schreiberhuber@acin.tuwien.ac.at), Thomas Faeulhammer (faeulhammer@acin.tuwien.ac.at)
 * @date 2017
 * @brief
 *
 */

#pragma once
#include <opencv/cv.h>
#include <v4r/segmentation/plane_extractor.h>

namespace v4r {

template <typename PointT>
class PlaneExtractorTile : public PlaneExtractor<PointT> {
 protected:
  using PlaneExtractor<PointT>::cloud_;
  using PlaneExtractor<PointT>::normal_cloud_;
  using PlaneExtractor<PointT>::all_planes_;
  using PlaneExtractor<PointT>::plane_inliers_;
  using PlaneExtractor<PointT>::param_;

  void do_compute(const boost::optional<const Eigen::Vector3f> &search_axis) override;

 public:
  explicit PlaneExtractorTile(const PlaneExtractorParameter &p = PlaneExtractorParameter());

  virtual bool getRequiresNormals() const override {
    return param_.pointwiseNormalCheck_;
  }

 private:
  /**
   * @brief The PlaneMatrix struct
   * structure to store a symmetrical 3 by 3 matrix
   */
  struct PlaneMatrix {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    double xx = 0.;
    double xy = 0.;
    double xz = 0.;
    double yy = 0.;
    double yz = 0.;
    double zz = 0.;
    size_t nrPoints = 0;

    PlaneMatrix() = default;

    inline void operator+=(const PlaneMatrix &b) {
      sum += b.sum;
      xx += b.xx;
      xy += b.xy;
      xz += b.xz;
      yy += b.yy;
      yz += b.yz;
      zz += b.zz;
      nrPoints += b.nrPoints;
    }

    inline PlaneMatrix operator+(const PlaneMatrix &b) {
      PlaneMatrix a(*this);
      a += b;
      return a;
    }

    void addPoint(const Eigen::Vector3f &p) {
      sum += p.cast<double>();
      xx += p(0) * p(0);
      xy += p(0) * p(1);
      xz += p(0) * p(2);
      yy += p(1) * p(1);
      yz += p(1) * p(2);
      zz += p(2) * p(2);
      nrPoints++;
    }
  };

  /**
   * @brief The PlaneSegment struct
   * plane segment
   */
  struct PlaneSegment {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector4f plane;
    size_t nrInliers = 0;
  };

  // maybe supply this with a list of additional
  struct Plane {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector4f plane;
    size_t nrElements;
  };

  std::vector<PlaneMatrix> matrices;
  std::vector<PlaneMatrix> planeMatrices;
  std::vector<std::vector<PlaneSegment>> planes;
  std::vector<std::vector<Eigen::Vector3f>> centerPoints;
  cv::Mat patchIds;

  // big todo for speed: switch to Vector4f elements
  // http://eigen.tuxfamily.org/index.php?title=FAQ#Vectorization

  Eigen::Vector4f calcPlaneFromMatrix(const PlaneMatrix &mat) const;

  void replace(int from, int to, int maxIndex);

  cv::Mat getDebugImage(bool doNormalTest);

  size_t minAbsBlockInlier;
  size_t colsOfPatches;  ///< The dimensions of the downsampled image of patches
  size_t rowsOfPatches;  ///< The dimensions of the downsampled image of patches
  size_t maxId = 0;      ///< The highest used id of

  int allocateMemory();

  void calculatePlaneSegments(bool doNormalTest);

  // TODO: this uses up too much time... get rid of it
  void rawPatchClustering();

  void postProcessing1Direction(const int offsets[][2], bool doNormalTest, bool reverse, bool zTest);
  void postProcessing(bool doNormalTest, bool zTest);

  /**
   * @brief zBuffer
   * This buffer contains the distance of a point to the assumed plane.
   * Only used when doZTest is set to true
   */
  cv::Mat zBuffer;

  /**
   * @brief thresholdsBuffer
   * Stores the thresholds for the according patches:
   * channel1 maxBlockDistance
   * channel2 minCosBlockAngle
   * channel3 maxInlierDistance
   * channel4 minCosAngle
   */
  std::vector<std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>>> thresholdsBuffer;

  float maxBlockAngle;     ///< The maximum angle that is allowed between two adjacent blocks to be able to connect them
  float minCosBlockAngle;  ///< The cos of this block angle
  float (*minCosBlockAngleFunc)(float depth) = nullptr;  ///< minCosBlockAngleFunc
  float maxAngle;     ///< The angle the normal vector of a pointer is never allowed to be off of a patch before getting
                      /// discarded.
  float minCosAngle;  ///< The cos of this angle
  float (*minCosAngleFunc)(float depth) = nullptr;         ///< minCosAngleFunc
  float (*maxInlierDistFunc)(float depth) = nullptr;       ///< maxInlierDistFunc
  float (*maxInlierBlockDistFunc)(float depth) = nullptr;  ///< inlierBlockDistanceFunc

  // Some parameters for maximum
  cv::Mat segmentation;
  std::vector<Plane> resultingPlanes;

  cv::Mat debug;

  /**
   * @brief generateColorCodedTexture
   * @return
   * a rgb image of the segmentation result
   */
  cv::Mat generateColorCodedTexture() const;

  /**
   * @brief generateColorCodedTextureDebug
   * @return
   */
  cv::Mat generateColorCodedTextureDebug() const;

 public:
  void setMaxAngle(float angle) {
    maxAngle = angle;
    minCosAngle = cos(maxAngle);
  }

  void setMaxBlockAngle(float angle) {
    maxBlockAngle = angle;
    minCosBlockAngle = cos(maxBlockAngle);
  }

  typedef std::shared_ptr<PlaneExtractorTile<PointT>> Ptr;
  typedef std::shared_ptr<PlaneExtractorTile<PointT> const> ConstPtr;
};
}  // namespace v4r
