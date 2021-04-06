#include <glog/logging.h>

#include <pcl/common/angles.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud_pointvector.h>
#include <pcl/pcl_config.h>
#include <pcl_1_8/keypoints/uniform_sampling.h>
#include <boost/algorithm/string.hpp>
#include <pcl/impl/instantiate.hpp>

#include <v4r/common/downsampler.h>
#include <v4r/common/greedy_local_clustering.h>
#include <v4r/common/point_types.h>

namespace v4r {

namespace po = boost::program_options;

void DownsamplerParameter::init(boost::program_options::options_description& desc, const std::string& section_name) {
  desc.add_options()((section_name + ".resolution").c_str(), po::value<float>(&resolution_)->default_value(resolution_),
                     "resolution of the downsampled point cloud in meter");
  desc.add_options()(
      (section_name + ".advanced_angular_distance_thershold").c_str(),
      po::value<float>(&advanced_angular_distance_thershold_)->default_value(advanced_angular_distance_thershold_),
      "Parameter of advanced downsampling method. After local clustering was performed inside an octree leaf, the "
      "clusters that have too few points (relative to the largest cluster) are discarded.");
  desc.add_options()((section_name + ".method").c_str(), po::value<Method>(&method_)->default_value(method_),
                     "downsampling method.");
}

std::istream& operator>>(std::istream& in, DownsamplerParameter::Method& t) {
  std::string token;
  in >> token;
  boost::to_upper(token);
  if (token == "UNIFORM")
    t = DownsamplerParameter::Method::UNIFORM;
  else if (token == "ADVANCED")
    t = DownsamplerParameter::Method::ADVANCED;
  else if (token == "TWO_LEVEL_ADVANCED")
    t = DownsamplerParameter::Method::TWO_LEVEL_ADVANCED;
  else if (token == "VOXEL")
    t = DownsamplerParameter::Method::VOXEL;
  else
    in.setstate(std::ios_base::failbit);
  return in;
}

std::ostream& operator<<(std::ostream& out, const DownsamplerParameter::Method& t) {
  switch (t) {
    case DownsamplerParameter::Method::UNIFORM:
      out << "UNIFORM";
      break;
    case DownsamplerParameter::Method::ADVANCED:
      out << "ADVANCED";
      break;
    case DownsamplerParameter::Method::TWO_LEVEL_ADVANCED:
      out << "TWO_LEVEL_ADVANCED";
      break;
    case DownsamplerParameter::Method::VOXEL:
      out << "VOXEL";
      break;
    default:
      out.setstate(std::ios_base::failbit);
  }
  return out;
}

template <typename PointT>
struct PointClusteringPolicy {
  using Object = PointT;
  using Cluster = pcl::CentroidPoint<PointT>;

  const float angular_distance_thershold_;

  /// \param[in] angular_distance_threshold if angular distance between point normal and cluster normal is below this
  ///                                       threshold (degrees), the point is merged into the cluster
  explicit PointClusteringPolicy(float angular_distance_thershold)
  : angular_distance_thershold_(std::cos(pcl::deg2rad(angular_distance_thershold))) {}

  bool similarToCluster(const Object& object, const Cluster& cluster) const {
    PointT centroid;
    cluster.get(centroid);
    return centroid.getNormalVector4fMap().dot(object.getNormalVector4fMap()) > angular_distance_thershold_;
  }

  void addToCluster(const Object& object, Cluster& cluster) const {
    cluster.add(object);
  }
};

namespace {

template <class PointT>
typename pcl::PointCloud<PointT>::Ptr doAdvancedDownsampling(const typename pcl::PointCloud<PointT>::ConstPtr& input,
                                                             float resolution, float angular_distance) {
  typename pcl::PointCloud<PointT>::Ptr downsampled(new pcl::PointCloud<PointT>);
  pcl::octree::OctreePointCloudPointVector<PointT> octree(resolution);
  octree.setInputCloud(input);
  octree.addPointsFromInputCloud();
#if PCL_VERSION_COMPARE(>=, 1, 9, 0)
  for (auto iter = octree.leaf_depth_begin(); iter != octree.leaf_depth_end(); ++iter) {
#else
  for (auto iter = octree.leaf_begin(); iter != octree.leaf_end(); ++iter) {
#endif
    auto& container = iter.getLeafContainer();
    GreedyLocalClustering<PointClusteringPolicy<PointT>> glc(angular_distance);
    for (const auto& idx : container.getPointIndicesVector()) {
      const auto& pt = input->at(idx);
      if (std::isnan(pt.normal_x) || std::isnan(pt.normal_y) || std::isnan(pt.normal_z))
        continue;  // skip points with NaN normals
      glc.add(pt);
    }
    auto clusters = glc.getClusters();
    std::sort(clusters.begin(), clusters.end(),
              [](const auto& c1, const auto& c2) { return c1.getSize() > c2.getSize(); });
    for (const auto& cluster : clusters) {
      PointT centroid;
      cluster.get(centroid);
      downsampled->push_back(centroid);
    }
  }
  return downsampled;
}

}  // namespace

template <typename PointT>
auto downsampleAdvanced(const typename pcl::PointCloud<PointT>::ConstPtr input, const DownsamplerParameter& param)
    -> std::enable_if_t<pcl::traits::has_normal<PointT>::value, typename pcl::PointCloud<PointT>::Ptr> {
  auto level0 = doAdvancedDownsampling<PointT>(input, param.resolution_, param.advanced_angular_distance_thershold_);

  if (param.method_ == DownsamplerParameter::Method::ADVANCED) {
    return level0;
  }

  // in Vidal et al 2018 paper they mention two step downsample process to reduce influence of flat surfaces
  // (2.2.1 they mention non-relevant surface characteristics bigger than voxel size)
  // this is not exactly how they do it, but it should be similar.
  return doAdvancedDownsampling<PointT>(level0, param.resolution_ * 2.f,
                                        param.advanced_angular_distance_thershold_ * 0.55f);
}

template <typename PointT>
auto downsampleAdvanced(const typename pcl::PointCloud<PointT>::ConstPtr /* input */, const DownsamplerParameter &
                        /* param */)
    -> std::enable_if_t<!pcl::traits::has_normal<PointT>::value, typename pcl::PointCloud<PointT>::Ptr> {
  LOG(ERROR) << "Advanced downsampling is not supported if point type does not have normal field";
  throw std::runtime_error("Advanced downsampling is not supported if point type does not have normal field");
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr downsampleVoxel(const typename pcl::PointCloud<PointT>::ConstPtr input,
                                                      const DownsamplerParameter& param) {
  typename pcl::PointCloud<PointT>::Ptr downsampled(new pcl::PointCloud<PointT>);
  pcl::VoxelGrid<PointT> grid;
  grid.setInputCloud(input);
  grid.setLeafSize(param.resolution_, param.resolution_, param.resolution_);
  grid.setDownsampleAllData(true);
  grid.filter(*downsampled);
  return downsampled;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr downsampleUniform(const typename pcl::PointCloud<PointT>::ConstPtr input,
                                                        const DownsamplerParameter& param, std::vector<int>& indices) {
  pcl_1_8::UniformSampling<PointT> us;
  us.setRadiusSearch(param.resolution_);
  us.setInputCloud(input);
  pcl::PointCloud<int> sampled_indices;
  us.compute(sampled_indices);

  /// TODO make Uniform sampling store this directly into a std::vector
  indices.resize(sampled_indices.size());
  for (size_t i = 0; i < sampled_indices.size(); i++)
    indices[i] = sampled_indices[i];

  typename pcl::PointCloud<PointT>::Ptr downsampled(new pcl::PointCloud<PointT>);
  pcl::copyPointCloud(*input, indices, *downsampled);
  return downsampled;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr Downsampler::downsample(const typename pcl::PointCloud<PointT>::ConstPtr input) {
  indices_.reset(new std::vector<int>);
  switch (param_.method_) {
    case DownsamplerParameter::Method::VOXEL:
      return downsampleVoxel<PointT>(input, param_);
    case DownsamplerParameter::Method::ADVANCED:
    case DownsamplerParameter::Method::TWO_LEVEL_ADVANCED:
      return downsampleAdvanced<PointT>(input, param_);
    case DownsamplerParameter::Method::UNIFORM:
      return downsampleUniform<PointT>(input, param_, *indices_);
    default:
      LOG(ERROR) << "Unsupported downsampling method";
      return nullptr;
  }
}

#define PCL_INSTANTIATE_downsampleCloud(T)                                          \
  template typename pcl::PointCloud<T>::Ptr Downsampler::downsample<T>( \
      const typename pcl::PointCloud<T>::ConstPtr);

PCL_INSTANTIATE(downsampleCloud, V4R_PCL_XYZ_POINT_TYPES)

}  // namespace v4r
