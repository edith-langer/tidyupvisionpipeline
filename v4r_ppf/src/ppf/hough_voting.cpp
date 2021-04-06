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

#define _USE_MATH_DEFINES
#include <algorithm>
#include <cmath>
#include <cstring>

#include <glog/logging.h>

#include <ppf/hough_voting.h>

namespace ppf {

HoughVoting::HoughVoting(unsigned int num_point_bins, unsigned int num_angle_bins)
: num_point_bins_(num_point_bins), num_angle_bins_(num_angle_bins), num_bins_(num_point_bins_ * num_angle_bins_) {
  CHECK(num_point_bins > 0) << "Number of point bins should be positive";
  CHECK(num_angle_bins > 0) << "Number of angle bins should be positive";
  vote_accumulator_.reset(new VoteAccumulator[num_bins_]);
  angle_step_ = M_PI * 2 / num_angle_bins_;
  reset();
}

void HoughVoting::reset() {
  std::memset(vote_accumulator_.get(), 0, num_bins_ * sizeof(VoteAccumulator));
}

float HoughVoting::getAngleDiscretizationStep() const {
  return angle_step_;
}

void HoughVoting::castVote(const LocalCoordinate& lc) {
  if (lc.model_point_index1 >= num_point_bins_)
    throw std::runtime_error("invalid point index");
  auto angle_bin = angleToBinIndex(lc.rotation_angle);
  auto bin = lc.model_point_index1 * num_angle_bins_ + angle_bin;
  vote_accumulator_[bin] += 1;
}

HoughVoting::Peak HoughVoting::getPeak() const {
  return createPeak(getMaxIndex());
}

HoughVoting::Peak::Vector HoughVoting::extractPeaks(size_t max_num_peaks, VoteAccumulator min_votes) {
  CHECK(min_votes != 0) << "Minimum number of votes for peak extraction can not be zero";
  max_num_peaks = max_num_peaks > 0 ? max_num_peaks : num_bins_;
  Peak::Vector peaks;
  peaks.reserve(max_num_peaks);
  for (size_t i = 0; i < max_num_peaks; ++i) {
    auto max_index = getMaxIndex();
    if (vote_accumulator_[max_index] < min_votes)
      break;

    peaks.push_back(createPeak(max_index));

    // The rest of the block zeros out this cell and its row neighbors with less votes
    auto row_begin = (max_index / num_angle_bins_) * num_angle_bins_;

    // Advance index taking care of wrap-around, return votes at the index
    auto next = [this, row_begin](int& offset, int delta) -> VoteAccumulator& {
      offset = (offset + num_angle_bins_ + delta) % num_angle_bins_;
      return vote_accumulator_[row_begin + offset];
    };

    // Zero out cells in one direction
    auto zero_neighbors = [&](int direction) {
      auto neighbor_votes = peaks.back().votes;
      int j = max_index - row_begin;
      while (true) {
        auto& votes = next(j, direction);
        if (votes == 0 || votes > neighbor_votes)
          break;
        neighbor_votes = votes;
        votes = 0;
      }
    };

    vote_accumulator_[max_index] = 0;
    zero_neighbors(+1);
    zero_neighbors(-1);
  }
  return peaks;
}

HoughVoting::Peak HoughVoting::getRowPeak(unsigned int point) const {
  return createPeak(getMaxIndex(point * num_angle_bins_, (point + 1) * num_angle_bins_));
}

HoughVoting::Peak::Vector HoughVoting::getRowPeaks() const {
  Peak::Vector peaks;
  peaks.reserve(num_point_bins_);
  for (size_t i = 0; i < num_point_bins_; ++i)
    peaks.push_back(getRowPeak(i));
  return peaks;
}

HoughVoting::VoteAccumulator HoughVoting::getVotes(unsigned int point_bin, unsigned int angle_bin) const {
  if (point_bin >= num_point_bins_ || angle_bin >= num_angle_bins_)
    throw std::runtime_error("invalid bin index");
  return vote_accumulator_[point_bin * num_angle_bins_ + angle_bin];
}

unsigned int HoughVoting::angleToBinIndex(float angle) const {
  auto bin = static_cast<int>(std::floor(angle / angle_step_)) % static_cast<int>(num_angle_bins_);
  if (bin < 0)
    bin += num_angle_bins_;
  return bin;
}

size_t HoughVoting::getMaxIndex() const {
  return getMaxIndex(0, num_bins_);
}

size_t HoughVoting::getMaxIndex(unsigned int begin, unsigned int end) const {
  const auto ptr_begin = vote_accumulator_.get() + begin;
  const auto ptr_end = vote_accumulator_.get() + end;
  return begin + std::distance(ptr_begin, std::max_element(ptr_begin, ptr_end));
}

HoughVoting::Peak HoughVoting::createPeak(unsigned int bin_index) const {
  auto point_bin = bin_index / num_angle_bins_;
  auto angle_bin = bin_index - point_bin * num_angle_bins_;
  Peak peak;
  peak.lc.model_point_index1 = point_bin;
  peak.lc.rotation_angle = (0.5f + angle_bin) * angle_step_;
  peak.votes = vote_accumulator_[bin_index];
  return peak;
}

}  // namespace ppf
