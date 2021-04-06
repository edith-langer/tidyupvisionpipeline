#include <exception>

#include <glog/logging.h>
#include <pcl/common/angles.h>
#include <pcl/common/time.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <v4r/common/graph_geometric_consistency.h>
#include <v4r/common/topsort_pruning.h>
#include <v4r/geometry/geometry.h>
#include <v4r/geometry/normals.h>
#include <boost/format.hpp>
#include <boost/graph/biconnected_components.hpp>
#include <boost/graph/bron_kerbosch_all_cliques.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/iteration_macros.hpp>

#include <pcl/visualization/pcl_visualizer.h>

namespace po = boost::program_options;

struct MaxTimeAllowedReachedException : public std::exception {
  const char* what() const noexcept override {
    return "MaxTimeAllowedReachedException";
  }
};

namespace v4r {

struct CliqueVisitor {
  CliqueVisitor(std::vector<std::vector<size_t>>& max_cliques,
                double max_time_allowed = std::numeric_limits<double>::infinity())
  : max_cliques(max_cliques), max_time_allowed_(max_time_allowed) {
    time_elapsed_.reset();
  }
  template <typename Clique, typename Graph>
  void clique(const Clique& c, const Graph& /*g*/) {
    std::vector<size_t> current_clique;
    for (auto it = c.begin(); it != c.end(); ++it) {
      current_clique.emplace_back(*it);
    }
    max_cliques.push_back(std::move(current_clique));
    if (time_elapsed_.getTime() > max_time_allowed_) {
      throw MaxTimeAllowedReachedException();
    }
  };
  void reset() {
    max_cliques.clear();
  }
  std::vector<std::vector<size_t>>& max_cliques;
  double max_time_allowed_;
  pcl::StopWatch time_elapsed_;
};

template <typename PointModelT, typename PointSceneT>
void GraphGeometricConsistencyGrouping<PointModelT, PointSceneT>::filterCorrespondences(
    const pcl::Correspondences& input_corrs, pcl::Correspondences& filtered_corrs, std::vector<int>& inlier_indices,
    Eigen::Matrix4f& trans) const {
  pcl::registration::CorrespondenceRejectorSampleConsensus<PointModelT> corr_rejector;
  corr_rejector.setMaximumIterations(param_.ransac_max_iterations_);
  corr_rejector.setInlierThreshold(param_.ransac_threshold_);
  corr_rejector.setInputSource(input_);
  corr_rejector.setInputTarget(scene_);
  corr_rejector.setSaveInliers(true);
  corr_rejector.getRemainingCorrespondences(input_corrs, filtered_corrs);
  corr_rejector.getInliersIndices(inlier_indices);
  trans = corr_rejector.getBestTransformation();
}

template <typename PointModelT, typename PointSceneT>
template <typename GraphViewType>
void GraphGeometricConsistencyGrouping<PointModelT, PointSceneT>::groupCorrespondences(
    std::vector<pcl::Correspondences>& model_instances, GraphGGCG& correspondence_graph,
    GraphViewType& correspondence_graph_view) {
  std::vector<size_t> model_instances_kept_indices;

  const typename boost::property_map<GraphViewType, edge_component_t>::type components =
      boost::get(edge_component, correspondence_graph_view);

  size_t n_cc = boost::biconnected_components(correspondence_graph_view, components);

  if (n_cc < 1)
    return;

  std::vector<std::set<size_t>> unique_vertices_per_cc(
      n_cc);                                     // stores vertices that belong to each connected component
  std::vector<GraphGGCG> biconnected_component;  // (sub-)graph for each connected component
  biconnected_component.reserve(n_cc);

  for (size_t i = 0; i < n_cc; i++) {
    GraphGGCG& g = correspondence_graph.create_subgraph();
    biconnected_component.push_back(std::move(g));
  }

  // get and store the vertices that belong to each connected component, and create a subgraph with this information
  typename boost::graph_traits<GraphViewType>::edge_iterator edgeIt, edgeEnd;
  boost::tie(edgeIt, edgeEnd) = boost::edges(correspondence_graph_view);
  for (; edgeIt != edgeEnd; ++edgeIt) {
    int c = components[*edgeIt];
    size_t source = boost::source(*edgeIt, correspondence_graph_view);
    size_t target = boost::target(*edgeIt, correspondence_graph_view);
    unique_vertices_per_cc[c].insert(source);
    unique_vertices_per_cc[c].insert(target);
  }

  for (size_t i = 0; i < n_cc; ++i) {
    for (const auto& v : unique_vertices_per_cc[i]) {
      boost::add_vertex(v, biconnected_component[i]);
    }
  }

  // Go through the connected components and decide whether to use CliqueGC or usualGC or ignore (cc_sizes[i] <
  // gc_threshold_)
  // Decision based on the number of vertices in the connected component and graph arboricity...
  std::vector<bool> cliques_computation_possible(n_cc, param_.use_graph_);
  for (size_t c = 0; c < n_cc; c++) {
    if (unique_vertices_per_cc[c].size() < param_.gc_threshold_)
      continue;

    const GraphGGCG& connected_graph = biconnected_component[c];

    std::set<size_t> correspondences_used;

    std::vector<std::vector<size_t>> correspondence_to_instance;
    if (param_.prune_by_CC_)
      correspondence_to_instance.resize(model_scene_corrs_->size());

    std::vector<std::vector<size_t>> cliques;
    if (cliques_computation_possible[c]) {
      CliqueVisitor visitor(cliques, param_.max_time_allowed_cliques_comptutation_);
      pcl::StopWatch t;
      try {
        boost::bron_kerbosch_all_cliques(connected_graph, visitor, param_.gc_threshold_);
      } catch (MaxTimeAllowedReachedException& e) {
        LOG(INFO) << "bron_kerbosch_all_cliques timed out (>" << param_.max_time_allowed_cliques_comptutation_
                  << "ms )";
      }
      if (cliques.size() < param_.min_cliques_to_proceed_) {
        cliques_computation_possible[c] = false;
      }
      VLOG(1) << "cliques found: " << cliques.size() << ", took: " << t.getTime() << " ms.";
    }

    if (cliques_computation_possible[c]) {
      // map vertex id from subgraph to full graph
      for (size_t i = 0; i < cliques.size(); i++) {
        for (size_t j = 0; j < cliques[i].size(); j++) {
          cliques[i][j] = connected_graph.m_global_vertex[cliques[i][j]];
        }
      }

      std::sort(cliques.begin(), cliques.end(),
                [](const std::vector<size_t>& a, const std::vector<size_t>& b) { return a.size() > b.size(); });

      if (!param_.cliques_big_to_small_)
        std::reverse(cliques.begin(), cliques.end());

      std::vector<size_t> taken_corresps(model_scene_corrs_->size(), 0);

      for (const auto& clique : cliques) {
        // create a new clique based on how many times the correspondences in *it clique were used
        std::vector<size_t> new_clique;
        new_clique.reserve(clique.size());

        size_t used = 0;
        for (size_t cc : clique) {
          if (taken_corresps[cc] < param_.max_taken_correspondence_) {
            new_clique.push_back(cc);  //(*it)
            used++;
          }
        }

        if (used >= param_.gc_threshold_) {
          new_clique.resize(used);

          pcl::Correspondences temp_corrs, filtered_corrs;
          temp_corrs.reserve(used);
          for (size_t j = 0; j < new_clique.size(); j++) {
            temp_corrs.push_back(model_scene_corrs_->operator[](new_clique[j]));
          }

          std::vector<int> inlier_indices;
          Eigen::Matrix4f trans;
          filterCorrespondences(temp_corrs, filtered_corrs, inlier_indices, trans);

          if ((filtered_corrs.size() >= param_.gc_threshold_) && !inlier_indices.empty()) {
            found_transformations_.push_back(trans);
            model_instances.push_back(filtered_corrs);

            // mark all inliers
            for (int idx : inlier_indices) {
              taken_corresps[new_clique[idx]]++;

              if (param_.prune_by_CC_) {
                correspondence_to_instance[new_clique[idx]].push_back(model_instances.size() - 1);
                correspondences_used.insert(new_clique[idx]);
              }
            }
          }
        }
      }
    } else {
      // use iterative gc for simple cases with lots of correspondences...
      LOG(WARNING) << "Correspondence grouping is too hard to solve it using cliques...";

      const auto& local_vertices = connected_graph.m_local_vertex;

      std::vector<size_t> consensus_set(local_vertices.size());
      std::vector<bool> taken_corresps(local_vertices.size(), false);

      if (!param_.prune_by_TS_) {
        typename boost::graph_traits<GraphGGCG>::vertex_iterator vertexIt, vertexEnd;
        boost::tie(vertexIt, vertexEnd) = boost::vertices(connected_graph);
        for (; vertexIt != vertexEnd; ++vertexIt) {
          if (boost::out_degree(*vertexIt, connected_graph) < (param_.gc_threshold_ - 1))
            taken_corresps[*vertexIt] = true;
        }
      }

      for (size_t i = 0; i < local_vertices.size(); i++) {
        if (taken_corresps[i])
          continue;

        size_t consensus_size = 0;
        consensus_set[consensus_size++] = i;

        for (size_t j = 0; j < local_vertices.size(); j++) {
          if (j == i || taken_corresps[j])
            continue;

          // Let's check if j fits into the current consensus set
          bool is_a_good_candidate = std::all_of(consensus_set.begin(), consensus_set.begin() + consensus_size,
                                                 [j, &connected_graph](size_t k) {
                                                   // check if edge (j, consensus_set[k] exists in the graph, if it
                                                   // does not, is_a_good_candidate = false!...
                                                   return boost::edge(j, k, connected_graph).second;
                                                 });

          if (is_a_good_candidate)
            consensus_set[consensus_size++] = j;
        }

        if (consensus_size < param_.gc_threshold_)
          continue;

        pcl::Correspondences temp_corrs, filtered_corrs;
        temp_corrs.reserve(consensus_size);

        for (size_t j = 0; j < consensus_size; j++)
          temp_corrs.push_back(model_scene_corrs_->at(connected_graph.m_global_vertex[consensus_set[j]]));

        std::vector<int> inlier_indices;
        Eigen::Matrix4f trans;
        filterCorrespondences(temp_corrs, filtered_corrs, inlier_indices, trans);

        if ((filtered_corrs.size() >= param_.gc_threshold_) && !inlier_indices.empty()) {
          found_transformations_.push_back(trans);
          model_instances.push_back(filtered_corrs);

          // mark all inliers
          for (int idx : inlier_indices) {
            taken_corresps[consensus_set[idx]] = true;

            if (param_.prune_by_CC_) {
              correspondence_to_instance[consensus_set[idx]].push_back(model_instances.size() - 1);
              correspondences_used.insert(consensus_set[idx]);
            }
          }
        }
      }
    }

    if (param_.prune_by_CC_) {
      size_t connected_graph_vertices_num = boost::num_vertices(connected_graph);
      GraphGGCGNoSubgraph connected_graph_used_edges(connected_graph_vertices_num);
      for (const auto& edge : connected_graph.m_local_edge) {
        boost::add_edge(edge.second.m_source, edge.second.m_target, connected_graph_used_edges);
      }

      typename boost::graph_traits<GraphGGCGNoSubgraph>::vertex_iterator vertexIt, vertexEnd;
      std::vector<typename boost::graph_traits<GraphGGCGNoSubgraph>::vertex_descriptor> to_be_removed;
      boost::tie(vertexIt, vertexEnd) = vertices(connected_graph_used_edges);
      for (; vertexIt != vertexEnd; ++vertexIt) {
        std::set<size_t>::const_iterator it;
        it = correspondences_used.find(*vertexIt);
        if (it == correspondences_used.end())
          to_be_removed.push_back(*vertexIt);
      }

      for (size_t i = 0; i < to_be_removed.size(); i++)
        clear_vertex(to_be_removed[i], connected_graph_used_edges);

      boost::vector_property_map<size_t> components2(connected_graph_vertices_num);
      size_t n_cc2 = boost::connected_components(connected_graph_used_edges, &components2[0]);

      std::vector<size_t> cc_sizes2(n_cc2, 0);
      for (size_t i = 0; i < connected_graph_vertices_num; i++)
        cc_sizes2[components2[i]]++;

      size_t ncc_overthres = 0;
      for (size_t i = 0; i < n_cc2; i++) {
        if (cc_sizes2[i] >= param_.gc_threshold_)
          ncc_overthres++;
      }
      // somehow now i need to do a Nonmax supression of the model_instances that are in the same CC
      // gather instances that were generated with correspondences found in a specific CC
      // correspondence_to_instance maps correspondences (vertices) to instance, we can use that i guess

      for (size_t internal_c = 0; internal_c < n_cc2; internal_c++) {
        // ignore if not enough vertices...
        size_t num_v_in_cc_tmp = cc_sizes2[internal_c];
        if (num_v_in_cc_tmp < param_.gc_threshold_)
          continue;

        std::set<size_t> instances_for_this_cc;
        {
          size_t local_vertex_idx = 0;
          for (const auto& global_vertex_descriptor : connected_graph.m_global_vertex) {
            if (components2[local_vertex_idx] == internal_c) {
              for (size_t k = 0; k < correspondence_to_instance[global_vertex_descriptor].size(); k++) {
                instances_for_this_cc.insert(correspondence_to_instance[global_vertex_descriptor][k]);
              }
            }
            local_vertex_idx++;
          }
        }

        std::set<size_t>::const_iterator it;
        size_t max_size = 0;
        for (it = instances_for_this_cc.begin(); it != instances_for_this_cc.end(); ++it) {
          if (max_size <= model_instances[*it].size()) {
            max_size = model_instances[*it].size();
            // max_idx = *it;
          }
        }

        float thres = 0.5f;
        for (it = instances_for_this_cc.begin(); it != instances_for_this_cc.end(); ++it) {
          if (model_instances[*it].size() > (max_size * thres))
            model_instances_kept_indices.push_back(*it);
        }
      }
    }
  }

  if (param_.prune_by_CC_) {
    for (size_t i = 0; i < model_instances_kept_indices.size(); i++) {
      model_instances[i] = model_instances[model_instances_kept_indices[i]];
      found_transformations_[i] = found_transformations_[model_instances_kept_indices[i]];
    }

    model_instances.resize(model_instances_kept_indices.size());
    found_transformations_.resize(model_instances_kept_indices.size());
  }

  // visualizeCorrespondences(*model_scene_corrs_);
}

template <typename PointModelT, typename PointSceneT>
void GraphGeometricConsistencyGrouping<PointModelT, PointSceneT>::buildCorrespondenceGraph(
    GraphGGCG& correspondence_graph) {

  float min_dist_for_cluster = param_.gc_size_ * param_.min_cluster_dist_multiplier_;

  for (size_t k = 0; k < model_scene_corrs_->size(); k++) {
    int scene_index_k = model_scene_corrs_->operator[](k).index_match;
    int model_index_k = model_scene_corrs_->operator[](k).index_query;
    const auto& scene_point_k = scene_->points[scene_index_k].getVector4fMap();
    const auto& model_point_k = input_->points[model_index_k].getVector4fMap();
    const auto& scene_normal_k = scene_->points[scene_index_k].getNormalVector4fMap();
    const auto& model_normal_k = input_->points[model_index_k].getNormalVector4fMap();

    for (size_t j = (k + 1); j < model_scene_corrs_->size(); j++) {
      int scene_index_j = model_scene_corrs_->operator[](j).index_match;
      int model_index_j = model_scene_corrs_->operator[](j).index_query;

      // same scene or model point constraint - 5th freq. meaning that  in the benchmarked test scene (CES)
      // this condition was most rarely true - but let's keep this condition at the first place given that
      // the condition is really simple to check to avoid unnecessary computation of more advanced checks below
      if (scene_index_j == scene_index_k || model_index_j == model_index_k)
        continue;

      const auto& scene_point_j = scene_->points[scene_index_j].getVector4fMap();
      const auto& model_point_j = input_->points[model_index_j].getVector4fMap();

      const Eigen::Vector4f d_scene = scene_point_k - scene_point_j;
      const Eigen::Vector4f d_model = model_point_k - model_point_j;

      float dist_scene_pts = d_scene.norm();
      float dist_model_pts = d_model.norm();

      // check distance consistency - 1st freq. (in the benchmarked test scene (CES) this condition was most often true)
      float distance = fabs(dist_model_pts - dist_scene_pts);
      if (distance > param_.gc_size_)
        continue;

      // minimum distance constraint - 4th freq.
      if ((dist_model_pts < min_dist_for_cluster) || (dist_scene_pts < min_dist_for_cluster))
        continue;

      if (param_.check_normals_orientation_) {
        const auto& scene_normal_j = scene_->points[scene_index_j].getNormalVector4fMap();
        const auto& model_normal_j = input_->points[model_index_j].getNormalVector4fMap();

        const auto tf = alignOrientedPointPairs<float>(model_point_k, model_normal_k, model_point_j, scene_point_k,
                                                       scene_normal_k, scene_point_j);

        const Eigen::Vector4f model_normal_k_aligned =
            tf * model_normal_k;  // the projected version model_normal_k_aligned_aligned should now aligned to
                                  // the projects surface normal of scene_k
        const Eigen::Vector4f model_normal_j_aligned =
            tf * model_normal_j;  // this is the final transformed normal we are interested in

        const float rotation_angle_diff_dotp_k = scene_normal_k.dot(model_normal_k_aligned);
        const float rotation_angle_diff_dotp_j = scene_normal_j.dot(model_normal_j_aligned);
        const float rotation_angle_diff_dotp_min = std::min(rotation_angle_diff_dotp_k, rotation_angle_diff_dotp_j);

        if (visualize_) {
          const float rotation_angle_diff = acos(rotation_angle_diff_dotp_min);
          VLOG(2) << "Surface normal difference (deg): " << pcl::rad2deg(rotation_angle_diff)
                  << ", confidence point pair a: " << model_scene_corrs_->operator[](k).weight
                  << ", confidence point pair b: " << model_scene_corrs_->operator[](j).weight;

          Eigen::Matrix4f tf_trans_model = Eigen::Matrix4f::Identity();
          tf_trans_model.block<4, 1>(0, 3) = -model_point_j;

          Eigen::Matrix4f tf_trans_scene = Eigen::Matrix4f::Identity();
          tf_trans_scene.block<4, 1>(0, 3) = -scene_point_j;

          visualizeCorrespondencePair(model_scene_corrs_->operator[](k), model_scene_corrs_->operator[](j), tf);
        }

        if (rotation_angle_diff_dotp_min < param_.thres_dot_distance_)
          continue;
      }

      boost::add_edge(k, j, correspondence_graph);
    }
  }
}

template <typename PointModelT, typename PointSceneT>
void GraphGeometricConsistencyGrouping<PointModelT, PointSceneT>::clusterCorrespondences(
    std::vector<pcl::Correspondences>& model_instances) {
  model_instances.clear();
  found_transformations_.clear();

  // for the old gc...
  pcl::CorrespondencesPtr sorted_corrs(new pcl::Correspondences(*model_scene_corrs_));
  std::sort(sorted_corrs->begin(), sorted_corrs->end(),
            [](const auto& i, const auto& j) { return i.distance < j.distance; });
  model_scene_corrs_ = sorted_corrs;

  CHECK(!model_scene_corrs_->empty()) << "Correspondences not set, please set them before calling this function!";

  GraphGGCG correspondence_graph(model_scene_corrs_->size());
  buildCorrespondenceGraph(correspondence_graph);

  if (param_.prune_by_TS_) {
    using PrunedGraphType = typename TopsortPruning<GraphGGCG>::PrunedGraphType;
    TopsortPruning<GraphGGCG> topsort_pruning(correspondence_graph, param_.gc_threshold_);
    PrunedGraphType pruned_correspondence_graph = topsort_pruning.getPrunedGraph();
    groupCorrespondences<PrunedGraphType>(model_instances, correspondence_graph, pruned_correspondence_graph);
  } else {
    groupCorrespondences<GraphGGCG>(model_instances, correspondence_graph, correspondence_graph);
  }
}

template <typename PointModelT, typename PointSceneT>
bool GraphGeometricConsistencyGrouping<PointModelT, PointSceneT>::recognize(
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>& transformations) {
  std::vector<pcl::Correspondences> model_instances;
  return this->recognize(transformations, model_instances);
}

template <typename PointModelT, typename PointSceneT>
bool GraphGeometricConsistencyGrouping<PointModelT, PointSceneT>::recognize(
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>& transformations,
    std::vector<pcl::Correspondences>& clustered_corrs) {
  transformations.clear();
  if (!this->initCompute()) {
    PCL_ERROR(
        "[GraphGeometricConsistencyGrouping::recognize()] Error! Model cloud or Scene cloud not set, please set them "
        "before calling again this function.\n");
    return (false);
  }

  clusterCorrespondences(clustered_corrs);

  transformations = found_transformations_;

  this->deinitCompute();
  return true;
}

void GraphGeometricConsistencyGroupingParameter::init(boost::program_options::options_description& desc,
                                                      const std::string& section_name) {
  desc.add_options()((section_name + ".size_thresh").c_str(),
                     po::value<size_t>(&gc_threshold_)->default_value(gc_threshold_),
                     "Minimum cluster size. At least 3 correspondences are needed to compute the 6DOF pose ");
  desc.add_options()((section_name + ".size").c_str(),
                     po::value<float>(&gc_size_)->default_value(gc_size_, boost::str(boost::format("%.2e") % gc_size_)),
                     "Resolution of the consensus set used to cluster correspondences together ");
  desc.add_options()((section_name + ".thres_dot_distance").c_str(),
                     po::value<float>(&thres_dot_distance_)
                         ->default_value(thres_dot_distance_, boost::str(boost::format("%.2e") % thres_dot_distance_)),
                     " ");
  desc.add_options()((section_name + ".use_graph").c_str(), po::value<bool>(&use_graph_)->default_value(use_graph_),
                     " ");
  desc.add_options()((section_name + ".min_cluster_dist_multiplier").c_str(),
                     po::value<float>(&min_cluster_dist_multiplier_)
                         ->default_value(min_cluster_dist_multiplier_,
                                         boost::str(boost::format("%.2e") % min_cluster_dist_multiplier_)),
                     "this value times gc_size is the minimum distance between two point pairs (either from model or "
                     "scene) to allow them to be clustered together");
  desc.add_options()((section_name + ".max_taken_correspondences").c_str(),
                     po::value<size_t>(&max_taken_correspondence_)->default_value(max_taken_correspondence_), " ");
  desc.add_options()((section_name + ".cliques_big_to_small").c_str(),
                     po::value<bool>(&cliques_big_to_small_)->default_value(cliques_big_to_small_), " ");
  desc.add_options()((section_name + ".check_normals_orientation").c_str(),
                     po::value<bool>(&check_normals_orientation_)->default_value(check_normals_orientation_), " ");
  desc.add_options()((section_name + ".max_time_for_cliques_computation").c_str(),
                     po::value<double>(&max_time_allowed_cliques_comptutation_)
                         ->default_value(max_time_allowed_cliques_comptutation_, "100.0"),
                     " max time allowed for finding maximal cliques procedure during grouping correspondences");
  desc.add_options()(
      (section_name + ".min_cliques_to_proceed").c_str(),
      po::value<size_t>(&min_cliques_to_proceed_)->default_value(min_cliques_to_proceed_, "10000"),
      "if finding maximal cliques procedure returns less than this defined value before timeout, "
      "correspondences will be no longer computed by this graph based approach but by the simpler greedy "
      "correspondence grouping algorithm");
  desc.add_options()((section_name + ".ransac_threshold").c_str(),
                     po::value<float>(&ransac_threshold_)
                         ->default_value(ransac_threshold_, boost::str(boost::format("%.2e") % ransac_threshold_)),
                     "maximum inlier threshold for RANSAC used for correspondence rejection");
  desc.add_options()(
      (section_name + ".ransac_max_iterations").c_str(),
      po::value<int>(&ransac_max_iterations_)->default_value(ransac_max_iterations_),
      "maximum iterations for RANSAC used for correspondence rejection (0... to disable correspondence rejection)");
  desc.add_options()((section_name + ".prune_by_CC").c_str(),
                     po::value<bool>(&prune_by_CC_)->default_value(prune_by_CC_), " ");
  desc.add_options()((section_name + ".prune_by_TS").c_str(),
                     po::value<bool>(&prune_by_TS_)->default_value(prune_by_TS_),
                     "remove vertices with degree smaller than (gc_threshold - 1), topological sort algorithm");
}

template <typename PointModelT, typename PointSceneT>
void GraphGeometricConsistencyGrouping<PointModelT, PointSceneT>::visualizeCorrespondencePair(
    const pcl::Correspondence& ck, const pcl::Correspondence& cj, const Eigen::Matrix4f& alignment_transform) {
  int scene_index_k = ck.index_match;
  int model_index_k = ck.index_query;

  int scene_index_j = cj.index_match;
  int model_index_j = cj.index_query;

  const auto& scene_point_k = scene_->points[scene_index_k].getVector4fMap();
  const auto& model_point_k = input_->points[model_index_k].getVector4fMap();

  const auto& scene_point_j = scene_->points[scene_index_j].getVector4fMap();
  const auto& model_point_j = input_->points[model_index_j].getVector4fMap();

  static pcl::visualization::PCLVisualizer::Ptr vis;
  static int vp1, vp2, vp3;
  if (!vis) {
    vis.reset(new pcl::visualization::PCLVisualizer);
    vis->createViewPort(0, 0, 0.33, 1, vp1);
    vis->createViewPort(0.33, 0, 0.66, 1, vp2);
    vis->createViewPort(0.66, 0, 1, 1, vp3);
  }
  vis->removeAllPointClouds(vp1);
  vis->removeAllPointClouds(vp2);
  vis->removeAllPointClouds(vp3);

  vis->addCoordinateSystem(0.4, "vp1_co", vp1);
  vis->addCoordinateSystem(0.4, "vp2_co", vp2);
  vis->addCoordinateSystem(0.4, "vp3_co", vp3);
  vis->removeAllShapes(vp1);
  vis->removeAllShapes(vp2);
  vis->removeAllShapes(vp3);
  vis->addPointCloud<PointSceneT>(scene_, "scene", vp1);

  typename pcl::PointCloud<PointModelT>::Ptr model_aligned(new pcl::PointCloud<PointModelT>);
  pcl::transformPointCloudWithNormals(*input_, *model_aligned, alignment_transform);
  vis->addPointCloud<PointModelT>(model_aligned, "model", vp2);

  vis->addSphere(scene_->points[scene_index_k], 0.005, 1, 0, 0, "scene_point_k", vp1);
  vis->addSphere(scene_->points[scene_index_j], 0.005, 0, 1, 0, "scene_point_j", vp1);
  vis->addSphere(model_aligned->points[model_index_k], 0.005, 0, 0, 1, "model_point_k", vp2);
  vis->addSphere(model_aligned->points[model_index_j], 0.005, 0, 1, 1, "model_point_j", vp2);

  {
    typename pcl::PointCloud<PointSceneT>::Ptr scene_pts(new pcl::PointCloud<PointSceneT>);
    scene_pts->resize(2);
    scene_pts->points[0] = scene_->points[scene_index_k];
    scene_pts->points[1] = scene_->points[scene_index_j];
    vis->addPointCloudNormals<PointSceneT>(scene_pts, 1, 0.04f, "scene_normals", vp1);

    typename pcl::PointCloud<PointModelT>::Ptr model_pts(new pcl::PointCloud<PointModelT>);
    model_pts->resize(2);
    model_pts->points[0] = model_aligned->points[model_index_k];
    model_pts->points[1] = model_aligned->points[model_index_j];
    pcl::PointCloud<pcl::Normal>::Ptr model_normal_pts(new pcl::PointCloud<pcl::Normal>);
    vis->addPointCloudNormals<PointModelT>(model_pts, 1, 0.04f, "model_normals", vp2);
  }

  {
    typename pcl::PointCloud<PointSceneT>::Ptr scene_pts(new pcl::PointCloud<PointSceneT>);
    scene_pts->resize(2);
    scene_pts->points[0].getVector4fMap() = scene_point_k;
    scene_pts->points[1].getVector4fMap() = scene_point_j;
    vis->addPointCloudNormals<PointSceneT>(scene_pts, 1, 0.04f, "scene_normals_vp3", vp3);
    vis->addSphere(scene_pts->points[0], 0.005, 1, 0, 0, "scene_point_k_vp3", vp3);
    vis->addSphere(scene_pts->points[1], 0.005, 0, 1, 0, "scene_point_j_vp3", vp3);

    typename pcl::PointCloud<PointModelT>::Ptr model_pts(new pcl::PointCloud<PointModelT>);
    model_pts->resize(2);
    model_pts->points[0].getVector4fMap() = model_point_k;
    model_pts->points[1].getVector4fMap() = model_point_j;
    pcl::transformPointCloudWithNormals(*model_pts, *model_pts, alignment_transform);
    vis->addSphere(model_pts->points[0], 0.005, 0, 0, 1, "model_point_k_vp3_rot_rot", vp3);
    vis->addSphere(model_pts->points[1], 0.005, 0, 1, 1, "model_point_j_vp3_rot_rot", vp3);
    vis->addPointCloudNormals<PointModelT>(model_pts, 1, 0.04f, "model_normals_vp3_rot_rot", vp3);
  }

  vis->spin();
}

template class GraphGeometricConsistencyGrouping<pcl::PointNormal, pcl::PointNormal>;
template class GraphGeometricConsistencyGrouping<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal>;
}  // namespace v4r
