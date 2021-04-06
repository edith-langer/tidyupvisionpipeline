/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#pragma once

#include <pcl/point_types.h>
#include <pcl/recognition/cg/correspondence_grouping.h>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/subgraph.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

namespace v4r {
struct GraphGeometricConsistencyGroupingParameter {
  size_t gc_threshold_ = 5;  ///< Minimum cluster size. At least 3 correspondences are needed to compute the 6DOF pose
  float gc_size_ =
      0.015f;  ///< Resolution of the consensus set used to cluster correspondences together. If the difference in
               /// distance between model keypoints and scene keypoints of a pair of correspondence is greater than
  /// this threshold, the correspondence pair will not be connected.
  float thres_dot_distance_ = 0.8f;
  bool use_graph_ = true;
  float min_cluster_dist_multiplier_ = 1.f;  ///< this value times gc_size is the minimum distance between two point
                                             ///< pairs (either from model or scene) to allow them to be clustered
                                             ///< together
  size_t max_taken_correspondence_ = 5;
  bool cliques_big_to_small_ = true;
  bool check_normals_orientation_ = true;
  double max_time_allowed_cliques_comptutation_ = 100.;  ///< max time allowed for finding maximal cliques procedure
                                                         ///< during grouping correspondences
  size_t min_cliques_to_proceed_ =
      10000;  ///< if finding maximal cliques procedure returns less than this defined value before
              ///< timeout, correspondences will be no longer computed by this graph based approach
              ///< but by the simpler greedy correspondence grouping algorithm
  float ransac_threshold_ = 0.015f;    ///< maximum inlier threshold for RANSAC used for correspondence rejection
  int ransac_max_iterations_ = 10000;  ///< maximum iterations for RANSAC used for correspondence rejection (0... to
                                       ///< disable correspondence rejection)
  bool prune_by_CC_ = false;
  bool prune_by_TS_ =
      true;  ///< remove vertices with degree smaller than (gc_threshold - 1), topological sort algorithm

  /**
   * @brief init parameters
   * @param command_line_arguments (according to Boost program options library)
   * @param section_name section name of program options
   */
  void init(boost::program_options::options_description &desc, const std::string &section_name = "cg");
};

template <typename PointModelT, typename PointSceneT>
class GraphGeometricConsistencyGrouping : public pcl::CorrespondenceGrouping<PointModelT, PointSceneT> {
 private:
  using pcl::CorrespondenceGrouping<PointModelT, PointSceneT>::input_;
  using pcl::CorrespondenceGrouping<PointModelT, PointSceneT>::scene_;
  using pcl::CorrespondenceGrouping<PointModelT, PointSceneT>::model_scene_corrs_;

  struct edge_component_t {
    typedef boost::edge_property_tag kind;
  } edge_component;

  using GraphGGCGNoSubgraph =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, size_t,
                            boost::property<edge_component_t, std::size_t, boost::property<boost::edge_index_t, int>>>;
  using GraphGGCG = boost::subgraph<GraphGGCGNoSubgraph>;

  /** \brief Transformations found by clusterCorrespondences method. */
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> found_transformations_;

  template <typename GraphViewType>
  void groupCorrespondences(std::vector<pcl::Correspondences> &model_instances, GraphGGCG &correspondence_graph,
                            GraphViewType &correspondence_graph_view);
  void buildCorrespondenceGraph(GraphGGCG &correspondence_graph);
  void clusterCorrespondences(std::vector<pcl::Correspondences> &model_instances);

  /**
   * @brief filter correspondences set using RANSAC
   * @param input_corrs Input correspondences to check
   * @param filtered_corrs Correspondences that are consistent with respect to RANSAC parameters
   * @param inlier_indices Inlier indices
   * @param trans Best transformation
   */
  void filterCorrespondences(const pcl::Correspondences &input_corrs, pcl::Correspondences &filtered_corrs,
                             std::vector<int> &inlier_indices, Eigen::Matrix4f &trans) const;

  /**
   * @brief visualize correspondence pair
   * @param ck first correspondence
   * @param cj second correspondence
   * @param alignment_transform transform aligning the model to the scene
   */
  void visualizeCorrespondencePair(const pcl::Correspondence &ck, const pcl::Correspondence &cj,
                                   const Eigen::Matrix4f &alignment_transform);

  bool visualize_ = false;  ///< if true, visualizes correspondence pair

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  GraphGeometricConsistencyGroupingParameter param_;

  /** \brief Constructor */
  GraphGeometricConsistencyGrouping(
      const GraphGeometricConsistencyGroupingParameter &p = GraphGeometricConsistencyGroupingParameter())
  : pcl::CorrespondenceGrouping<PointModelT, PointSceneT>() {
    param_ = p;
  }

  /** \brief The main function, recognizes instances of the model into the scene set by the user.
   *
   * \param[out] transformations a vector containing one transformation matrix for each instance of the model
   * recognized into the scene.
   *
   * \return true if the recognition had been successful or false if errors have occurred.
   */
  bool recognize(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &transformations);

  /** \brief The main function, recognizes instances of the model into the scene set by the user.
   *
   * \param[out] transformations a vector containing one transformation matrix for each instance of the model
   * recognized into the scene.
   * \param[out] clustered_corrs a vector containing the correspondences for each instance of the model found within
   * the input data (the same output of clusterCorrespondences).
   *
   * \return true if the recognition had been successful or false if errors have occurred.
   */
  bool recognize(std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> &transformations,
                 std::vector<pcl::Correspondences> &clustered_corrs);

  void setVisualizeCorrespondencePairs(bool vis = true) {
    visualize_ = vis;
  }

  typedef std::shared_ptr<GraphGeometricConsistencyGrouping<PointModelT, PointSceneT>> Ptr;
  typedef std::shared_ptr<const GraphGeometricConsistencyGrouping<PointModelT, PointSceneT>> ConstPtr;
};
}  // namespace v4r
