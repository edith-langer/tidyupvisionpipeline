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

#include <utility>
#include <vector>

namespace v4r {

namespace detail {

/// An utility metafunction that maps a sequence of any types to the type void.
/// TODO: switch to std::void_t when C++17 is enabled.
template <class...>
using void_t = void;

/// An utility metafunction to check if a given Policy type has createCluster() method.
template <typename Policy, typename = void_t<>>
struct has_create_cluster : std::false_type {};

template <typename Policy>
struct has_create_cluster<Policy, void_t<decltype(std::declval<const Policy>().createCluster(
                                      std::declval<const typename Policy::Object&>()))>> : std::true_type {};

/// An utility metafunction to check if a given Policy type has addToCluster() method.
template <typename Policy, typename = void_t<>>
struct has_add_to_cluster : std::false_type {};

template <typename Policy>
struct has_add_to_cluster<
    Policy, void_t<decltype(std::declval<const Policy>().addToCluster(std::declval<const typename Policy::Object&>(),
                                                                      std::declval<typename Policy::Cluster&>()))>>
: std::true_type {};

/// An utility metafunction to check if a given Policy has Object type that is convertible to Cluster type.
template <typename Policy>
struct convertible_object_to_cluster : std::is_convertible<typename Policy::Object, typename Policy::Cluster> {};

/// A helper function to create a new cluster from an object (using a given policy).
/// The following pseudo-code summarizes what the function does:
///   if Policy::createCluster method exist
///     return Policy::createCluster(object)
///   elif Policy::Object is convertible to Policy::Cluster
///     return object
///   else
///     default-construct cluster
///     execute Policy::addToCluster(object, cluster)
///     return cluster
template <typename Policy>
auto createCluster(const Policy& policy, const typename Policy::Object& object)
    -> std::enable_if_t<has_create_cluster<Policy>::value, typename Policy::Cluster> {
  return policy.createCluster(object);
}

template <typename Policy>
auto createCluster(const Policy&, const typename Policy::Object& object)
    -> std::enable_if_t<!has_create_cluster<Policy>::value && convertible_object_to_cluster<Policy>::value,
                        typename Policy::Cluster> {
  return object;
}

template <typename Policy>
auto createCluster(const Policy& policy, const typename Policy::Object& object)
    -> std::enable_if_t<!has_create_cluster<Policy>::value && !convertible_object_to_cluster<Policy>::value,
                        typename Policy::Cluster> {
  typename Policy::Cluster cluster{};
  policy.addToCluster(object, cluster);
  return cluster;
}

template <typename Policy>
auto addToCluster(const Policy& policy, const typename Policy::Object& object, typename Policy::Cluster& cluster)
    -> std::enable_if_t<has_add_to_cluster<Policy>::value> {
  policy.addToCluster(object, cluster);
}

template <typename Policy>
auto addToCluster(const Policy&, const typename Policy::Object&, typename Policy::Cluster&)
    -> std::enable_if_t<!has_add_to_cluster<Policy>::value> {}

/// A meta-function that generates a vector type to hold a collection of clusters for a given Policy using a custom
/// allocator, if requested.
/// The following pseudo-code summarizes what the function does:
///   if Policy::ClusterAllocator typedef exists
///     return std::vector<Policy::Cluster, Policy::ClusterAllocator>
///   else
///     return std::vector<Policy::Cluster>
template <typename Policy, typename = void_t<>>
struct Clusters {
  using type = std::vector<typename Policy::Cluster>;
};

template <typename Policy>
struct Clusters<Policy, void_t<typename Policy::ClusterAllocator>> {
  using type = std::vector<typename Policy::Cluster, typename Policy::ClusterAllocator>;
};

}  // namespace detail

/// Generic algorithm that implements greedy local clustering of objects.
///
/// The template parameter is a policy class that defines the type of objects to be clustered, as well as several basic
/// operations needed to perform clustering. Specifically, the policy should define:
///  * `Object`
///    Type of objects that are to be clustered.
///  * `Cluster`
///    Type that represents a cluster of objects.
///  * `ClusterAllocator` [optional]
///    An allocator to be used in a vector of Cluster objects. If not present, the default std allocator is used. This
///    is needed e.g. in case when cluster type contains Eigen objects and thus needs special aligned allocator.
///  * `bool similarToCluster(const Object& object, const Cluster& cluster) const`
///    Method that checks whether a given object is similar enough to a given cluster (and should be merged into it).
///  * `void addToCluster(const Object& object, Cluster& cluster) const` [optional]
///    Method that adds a given object to a given cluster. If not present, adding objects to clusters is a no-op.
///  * `Cluster createCluster(const Object& object) const` [optional]
///    Method that creates a new cluster from a given object. If not present, the algorithm will do either of the
///    following:
///      - convert object to cluster (if the types are convertible)
///      - default-construct a new cluster and use `addToCluster()` to insert the object. Note that the default
///        constructor should perform some meaningful initialization of the cluster.
///
/// To instantiate the algorithm either pass it an instance of the policy class, or the arguments that can be used to
/// instantiate one. After that the objects can be added either one-by-one with or in bulk. The getClusters() method can
/// be used to retrieve current clusters at any point in time.
template <typename Policy>
class GreedyLocalClustering {
 public:
  using Object = typename Policy::Object;
  using Cluster = typename Policy::Cluster;
  using Clusters = typename detail::Clusters<Policy>::type;

  explicit GreedyLocalClustering(const Policy& policy) : policy_(policy) {}

  template <typename... Args>
  explicit GreedyLocalClustering(Args&&... args) : policy_{std::forward<Args>(args)...} {}

  void add(const Object& obj) {
    bool found_cluster = false;
    for (auto& cluster : clusters_) {
      if (policy_.similarToCluster(obj, cluster)) {
        detail::addToCluster(policy_, obj, cluster);
        found_cluster = true;
        break;
      }
    }
    if (!found_cluster) {
      clusters_.push_back(detail::createCluster(policy_, obj));
    }
  }

  const Clusters& getClusters() const {
    return clusters_;
  }

  template <typename Iterator>
  void add(Iterator begin, Iterator end) {
    for (auto iter = begin; iter != end; ++iter)
      add(*iter);
  }

 private:
  const Policy policy_;
  Clusters clusters_;
};

}  // namespace v4r
