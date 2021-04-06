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

#include <memory>
#include <vector>

#include <ppf/local_coordinate.h>

namespace ppf {

/// Specialized class for 2D Hough Voting in the local coordinate space.
class HoughVoting {
 public:
  using VoteAccumulator = uint16_t;

  /// A peak in the Hough voting space (LC and number of votes).
  struct Peak {
    LocalCoordinate lc;
    VoteAccumulator votes;
    using Vector = std::vector<Peak>;
  };

  /// Construct a new voting space of given dimensions.
  HoughVoting(unsigned int num_point_bins, unsigned int num_angle_bins);

  /// Set all cells to zero.
  void reset();

  /// Get discretization step in angle dimension (radians).
  float getAngleDiscretizationStep() const;

  /// Cast a vote for a given local coordinate.
  void castVote(const LocalCoordinate& lc);

  /// Get global peak in the voting space.
  /// Peak angle is the angle at the center of the peak cell.
  /// In case of a tie the first peak (row-major) is returned.
  Peak getPeak() const;

  /// Get peaks in the voting space.
  /// Each returned peak is a local maximum (in its row neighborhood). There may be multiple peaks from the same row.
  /// Peak angle is the angle at the center of the peak cell.
  /// The peaks are sorted in descending order by the number of votes. In case of a tie (all neighbors have the same
  /// number of votes) the left-most cell is considered to be a peak.
  /// \params[in] max_num_peaks return at most given number of peaks (0 means all)
  /// \params[in] min_votes do not consider cells with less than this number of votes (0 is invalid input)
  /// \warning This operation invalidates the vote accumulator!
  Peak::Vector extractPeaks(size_t max_num_peaks = 0, VoteAccumulator min_votes = 1);

  /// Get peak in a given row of the voting space.
  /// Peak angle is the angle at the center of the peak cell.
  /// In case of a tie the first peak is returned.
  Peak getRowPeak(unsigned int point) const;

  /// Get peak in each row of the voting space.
  /// Peak angle is the angle at the center of the peak cell.
  /// In case of a tie the first peak is returned.
  Peak::Vector getRowPeaks() const;

  /// Get number of votes cast into a given bin.
  VoteAccumulator getVotes(unsigned int point_bin, unsigned int angle_bin) const;

 private:
  /// Discretize angle into bin index.
  /// Takes into account angle periodicity.
  unsigned int angleToBinIndex(float angle) const;

  /// Get index of the max element in the entire vote accumulator.
  size_t getMaxIndex() const;

  /// Get index of the max element in a section [begin, end) of vote accumulator.
  /// Returned index is into the entire vote accumulator, not just section.
  size_t getMaxIndex(unsigned int begin, unsigned int end) const;

  /// Create an instance of Peak from a given bin in the vote accumulator.
  Peak createPeak(unsigned int bin_index) const;

  std::unique_ptr<VoteAccumulator[]> vote_accumulator_;
  unsigned int num_point_bins_;
  unsigned int num_angle_bins_;
  size_t num_bins_;
  float angle_step_;
};

}  // namespace ppf

