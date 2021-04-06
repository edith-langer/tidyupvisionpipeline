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
****************************************************************************
*/

/*
 * @author Adam Chudas
 * @date 18.05.18
 * @file
 * @brief Algorithm take graph G and threshold T
 * then removes from G vertices with degree smaller than T-1
 * until there are no such left.
 *
 * Remaining vertices are the only one
 * which have a chance to be in clique of size at least T
 * since every such a vertex has to have at least T-1 neighbours.
 *
 * Complexity is O(V+E), because algorithm uses only single Breadth First Search traversal.
 * At the beginning, it adds to G dummy vertex D connected with every vertex in G.
 * Then it starts with queue Q containing single vertex D.
 * If vertex is in Q it means that it is to be deleted (degree smaller than T-1).
 * In each loop we remove vertex S from front of Q and for each vertex V adjacent to S
 * we decrease degree of V by one and if its degree become smaller than T-1 we add it to Q.
 *
 */

#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/filtered_graph.hpp>
#include <boost/graph/subgraph.hpp>

namespace v4r {

template <typename T>
struct is_subgraph {
  static constexpr bool value = false;
};

template <typename T>
struct is_subgraph<boost::subgraph<T>> {
  static constexpr bool value = true;
};

/**
 * @class bfs_degree_visitor
 * @brief Breadth First Search visitor for topsort pruning
 * */
template <typename DegreeMap, typename ColorMap, typename GraphType>
class bfs_degree_visitor : public boost::default_bfs_visitor {
  using Vertex = typename boost::graph_traits<GraphType>::vertex_descriptor;
  using Edge = typename boost::graph_traits<GraphType>::edge_descriptor;

 public:
  /**
   *
   * @param[in] dmap iterator property map for degrees of vertices
   * @param[in] cmap iterator property map for colors of vertices
   * @param[in] dummy descriptor of dummy vertex D, from which BFS will start
   * @param[in] min_cq_size minimum size of clique
   * @param[in] g the graph
   */
  bfs_degree_visitor(DegreeMap dmap, ColorMap cmap, Vertex dummy, int min_cq_size, GraphType &g)
  : degreemap_(dmap), colormap_(cmap), dummy_vertex_(dummy), thresh_(min_cq_size - 1), g_(g) {}

  /**
   * @brief Called for each vertex in graph at the beginning, add edge to dummy vertex D
   */
  void initialize_vertex(Vertex s, const GraphType &g) {
    boost::add_edge(boost::vertex(s, g_), boost::vertex(dummy_vertex_, g_), g_);

    int deg = get_degree(s, g);

    boost::put(degreemap_, s, deg);
  }

  template <typename GraphT>
  typename std::enable_if<!is_subgraph<GraphT>::value, int>::type get_degree(Vertex s, const GraphT &g) {
    return g.out_edge_list(s).size();
  }

  template <typename GraphT>
  typename std::enable_if<is_subgraph<GraphT>::value, int>::type get_degree(Vertex s, const GraphT &g) {
    return g.m_graph.out_edge_list(s).size();
  }

  /**
   * @brief Called for each edge incident with vertex U, when U is removed from Q.
   * For each neighbour V of vertex U, decrease degree of V by one.
   * Mark V as black (already processed) if necessary so as to BFS would not add V to Q.
   * Mark V as white if V was previously marked as black but now became ready to be removed.
   */
  void examine_edge(Edge e, const GraphType &g) {
    Vertex v = boost::target(e, g);
    int deg = boost::get(degreemap_, v);
    deg--;
    boost::put(degreemap_, v, deg);

    auto color = boost::get(colormap_, v);

    if (deg >= thresh_ and color == boost::color_traits<GraphType>::white())
      boost::put(colormap_, boost::vertex(v, g_), boost::color_traits<GraphType>::black());
    else if (deg == thresh_ - 1 and color == boost::color_traits<GraphType>::black())
      boost::put(colormap_, boost::vertex(v, g_), boost::color_traits<GraphType>::white());
  }

 private:
  DegreeMap degreemap_;
  ColorMap colormap_;
  Vertex dummy_vertex_;
  int thresh_;
  GraphType &g_;
};

/**
 * @class TopsortPruning
 * @brief Wrappper class for running topsort pruning on given graph
 * @tparam GraphType
 */
template <class GraphType>
class TopsortPruning {
  using Vertex = typename boost::graph_traits<GraphType>::vertex_descriptor;
  using SetPtr = typename boost::shared_ptr<std::set<Vertex>>;

  struct vertex_predicate_s {
    bool operator()(const Vertex &v) const {
      return removed->find(v) == removed->end();
    }

    vertex_predicate_s() = default;
    vertex_predicate_s(const SetPtr &removed) : removed(removed) {}

    SetPtr removed;
  };

 public:
  using PrunedGraphType = boost::filtered_graph<GraphType, boost::keep_all, vertex_predicate_s>;

  /**
   * @param graph Given graph to be pruned, new single vertex linked to every vertex is added to the original graph
   * @param min_cq_size Minimum clique size to use as a parameter for pruning
   */
  TopsortPruning(GraphType &graph, int min_cq_size) : graph_(graph), min_cq_size_(min_cq_size), pruned_graph_(nullptr) {
    pruneGraphByTopsort();
  }

  /**
   * @return reference to pruned_graph, valid as long as containing instance of TopsortPruning class is alive
   */
  PrunedGraphType &getPrunedGraph() {
    return *pruned_graph_;
  }

 private:
  void pruneGraphByTopsort();

  GraphType &graph_;
  int min_cq_size_;
  boost::shared_ptr<PrunedGraphType> pruned_graph_;  ///< filtered graph being result of pruning the original graph
};

template <class GraphType>
inline void TopsortPruning<GraphType>::pruneGraphByTopsort() {
  Vertex dummy_vertex = boost::add_vertex(graph_);

  typedef typename boost::property_map<GraphType, boost::vertex_index_t>::type index_map;
  typedef std::vector<int> degree_vector;
  typedef boost::iterator_property_map<degree_vector::iterator, index_map> degree_map_t;

  degree_vector degree_storage(boost::num_vertices(graph_));
  index_map idx_map = get(boost::vertex_index, graph_);
  degree_map_t deg_map = boost::make_iterator_property_map(degree_storage.begin(), idx_map);

  typedef std::vector<boost::default_color_type> color_vector;
  typedef boost::iterator_property_map<color_vector::iterator, index_map> color_map_t;

  color_vector color_storage(boost::num_vertices(graph_), boost::default_color_type::white_color);
  color_map_t col_map = boost::make_iterator_property_map(color_storage.begin(), idx_map);

  bfs_degree_visitor<degree_map_t, color_map_t, GraphType> vis(deg_map, col_map, dummy_vertex, min_cq_size_, graph_);
  breadth_first_search(graph_, dummy_vertex, visitor(vis).color_map(col_map));

  typename boost::graph_traits<GraphType>::vertex_iterator vi, vi_end;
  SetPtr removed_set(new std::set<Vertex>);
  for (boost::tie(vi, vi_end) = boost::vertices(graph_); vi != vi_end; ++vi)
    if (boost::get(deg_map, *vi) < (int)min_cq_size_ - 1)
      removed_set->insert(*vi);
  removed_set->insert(dummy_vertex);

  vertex_predicate_s vertex_predicate(removed_set);
  pruned_graph_.reset(new PrunedGraphType(graph_, boost::keep_all{}, vertex_predicate));
}

}  // namespace v4r
