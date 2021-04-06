#include <glog/logging.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/normal_3d_omp.h>
#include <v4r/geometry/normals.h>
#include <boost/algorithm/string.hpp>
#include <numeric>
#include <pcl/impl/instantiate.hpp>
#include <pcl/io/pcd_io.h>

namespace po = boost::program_options;

namespace v4r {

void transformNormals(const pcl::PointCloud<pcl::Normal> &normals_cloud, pcl::PointCloud<pcl::Normal> &normals_aligned,
                      const Eigen::Matrix4f &transform) {
  normals_aligned.points.resize(normals_cloud.points.size());
  normals_aligned.width = normals_cloud.width;
  normals_aligned.height = normals_cloud.height;
  for (size_t k = 0; k < normals_cloud.points.size(); k++) {
    normals_aligned.points[k].getNormalVector4fMap() = transform * normals_cloud.points[k].getNormalVector4fMap();
    normals_aligned.points[k].curvature = normals_cloud.points[k].curvature;
  }
}

void transformNormals(const pcl::PointCloud<pcl::Normal> &normals_cloud, pcl::PointCloud<pcl::Normal> &normals_aligned,
                      const std::vector<int> &indices, const Eigen::Matrix4f &transform) {
  normals_aligned.points.resize(indices.size());
  normals_aligned.width = indices.size();
  normals_aligned.height = 1;
  for (size_t k = 0; k < indices.size(); k++) {
    normals_aligned.points[k].getNormalVector4fMap() =
        transform * normals_cloud.points[indices[k]].getNormalVector4fMap();
    normals_aligned.points[k].curvature = normals_cloud.points[indices[k]].curvature;
  }
}

std::istream &operator>>(std::istream &in, NormalEstimatorParameter::Method &nt) {
  std::string token;
  in >> token;
  boost::to_upper(token);
  if (token == "PCL_DEFAULT")
    nt = NormalEstimatorParameter::Method::PCL_DEFAULT;
  else if (token == "PCL_INTEGRAL_NORMAL")
    nt = NormalEstimatorParameter::Method::PCL_INTEGRAL_NORMAL;
  else if (token == "Z_ADAPTIVE")
    nt = NormalEstimatorParameter::Method::Z_ADAPTIVE;
  else
    in.setstate(std::ios_base::failbit);
  return in;
}

std::ostream &operator<<(std::ostream &out, const NormalEstimatorParameter::Method &nt) {
  switch (nt) {
    case NormalEstimatorParameter::Method::PCL_DEFAULT:
      out << "PCL_DEFAULT";
      break;
    case NormalEstimatorParameter::Method::PCL_INTEGRAL_NORMAL:
      out << "PCL_INTEGRAL_NORMAL";
      break;
    case NormalEstimatorParameter::Method::Z_ADAPTIVE:
      out << "Z_ADAPTIVE";
      break;
    default:
      out.setstate(std::ios_base::failbit);
  }
  return out;
}

void NormalEstimatorParameter::init(boost::program_options::options_description &desc,
                                    const std::string &section_name) {
  desc.add_options()((section_name + ".radius").c_str(), po::value<float>(&radius_)->default_value(radius_),
                     "support radius in meter for surface normal estimation.");
  desc.add_options()((section_name + ".use_omp").c_str(), po::value<bool>(&use_omp_)->default_value(use_omp_),
                     "if true, uses openmp for surface normal estimation.");
  desc.add_options()((section_name + ".smoothing_size").c_str(),
                     po::value<float>(&smoothing_size_)->default_value(smoothing_size_),
                     "smoothing size for surface normal estimation.");
  desc.add_options()((section_name + ".max_depth_change_factor").c_str(),
                     po::value<float>(&max_depth_change_factor_)->default_value(max_depth_change_factor_),
                     "depth change threshold for computing object borders for surface normal estimation.");
  desc.add_options()((section_name + ".use_depth_depended_smoothing").c_str(),
                     po::value<bool>(&use_depth_depended_smoothing_)->default_value(use_depth_depended_smoothing_),
                     "use depth depended smoothing for surface normal estimation.");
  desc.add_options()((section_name + ".kernel_size").c_str(), po::value<int>(&kernel_)->default_value(kernel_),
                     "kernel radius in pixel for surface normal estimation.");
  desc.add_options()((section_name + ".z_adaptive").c_str(), po::value<bool>(&adaptive_)->default_value(adaptive_),
                     "if true, adapts kernel radius with distance of point to camera  for surface normal estimation.");
  desc.add_options()((section_name + ".kappa").c_str(), po::value<float>(&kappa_)->default_value(kappa_),
                     "gradient for surface normal estimation.");
  desc.add_options()((section_name + ".d").c_str(), po::value<float>(&d_)->default_value(d_),
                     "constant for surface normal estimation.");
  desc.add_options()((section_name + ".kernel_radii").c_str(),
                     po::value<std::vector<int>>(&kernel_radius_)->multitoken(),
                     "Kernel radius for each 0.5 meter interval (e.g. if 8 elements, then 0-4m).");
  desc.add_options()((section_name + ".method").c_str(),
                     po::value<NormalEstimatorParameter::Method>(&method_)->default_value(method_),
                     "normal computation method (PCL_DEFAULT, PCL_INTEGRAL_NORMAL, Z_ADAPTIVE)");
}

template <typename PointT>
pcl::Normal computeNormal(const pcl::PointCloud<PointT> &cloud, const std::vector<int> &indices) {
  pcl::Normal n;
  n.curvature = n.normal_x = n.normal_y = n.normal_z = std::numeric_limits<float>::quiet_NaN();

  if (indices.size() >= 4) {
    Eigen::Vector4d mean;
    Eigen::Matrix3d cov;
    pcl::computeMeanAndCovarianceMatrix(cloud, indices, cov, mean);

    Eigen::Vector3d eigen_values;
    Eigen::Matrix3d eigen_vectors;
    pcl::eigen33(cov, eigen_vectors, eigen_values);

    n.getNormalVector3fMap() = eigen_vectors.col(0).cast<float>();

    double eigsum = eigen_values.sum();
    if (eigsum != 0.)
      n.curvature = fabs(eigen_values[0] / eigsum);
  }
  return n;
}

template <typename PointT>
pcl::PointCloud<pcl::Normal>::Ptr computeNormals(const typename pcl::PointCloud<PointT>::ConstPtr cloud,
                                                 const NormalEstimatorParameter &param,
                                                 const pcl::IndicesConstPtr indices) {
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::IndicesConstPtr _indices;

  bool fake_indices = false;
  if (!indices) {
    fake_indices = true;
    pcl::IndicesPtr indices_tmp(new std::vector<int>(cloud->width * cloud->height));
    std::iota(indices_tmp->begin(), indices_tmp->end(), 0);
    _indices = indices_tmp;
  } else {
    _indices = indices;
  }
  normals->resize(_indices->size());

  if (fake_indices) {
    normals->width = cloud->width;
    normals->height = cloud->height;
  } else {
    normals->width = _indices->size();
    normals->height = 1;
  }

  switch (param.method_) {
    case NormalEstimatorParameter::Method::PCL_DEFAULT: {
      typename pcl::NormalEstimation<PointT, pcl::Normal>::Ptr ne;

      if (param.use_omp_)
        ne.reset(new pcl::NormalEstimationOMP<PointT, pcl::Normal>);
      else
        ne.reset(new pcl::NormalEstimation<PointT, pcl::Normal>);

      ne->setRadiusSearch(param.radius_);
      ne->setInputCloud(cloud);
      ne->setIndices(_indices);
      ne->compute(*normals);
      break;
    }
    case NormalEstimatorParameter::Method::PCL_INTEGRAL_NORMAL: {
      pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
      ne.setNormalEstimationMethod(ne.COVARIANCE_MATRIX);
      ne.setMaxDepthChangeFactor(param.max_depth_change_factor_);
      ne.setNormalSmoothingSize(param.smoothing_size_);
      ne.setDepthDependentSmoothing(param.use_depth_depended_smoothing_);
      ne.setInputCloud(cloud);
      ne.setIndices(_indices);
      ne.compute(*normals);
      break;
    }
    case NormalEstimatorParameter::Method::Z_ADAPTIVE: {
#pragma omp parallel for if (param.use_omp_)
      for (size_t i = 0; i < _indices->size(); i++) {
        int idx = indices->operator[](i);
        const PointT &p = cloud->operator[](idx);
        pcl::Normal &n = normals->operator[](idx);

        if (!pcl::isFinite(p)) {
          n.normal_x = n.normal_y = n.normal_z = std::numeric_limits<float>::quiet_NaN();
          continue;
        }

        // get neighboring indices
        /// TODO support also unorganized point clouds
        int u = (*_indices)[i] % cloud->width;
        int v = (*_indices)[i] / cloud->width;

        int kernel_radius = param.kernel_;
        if (param.adaptive_) {
          int radius_id = std::min<int>((int)(param.kernel_radius_.size()) - 1,
                                        (int)(p.z * 2.f));  // *2 => every 0.5 meter another kernel radius
          kernel_radius = param.kernel_radius_[radius_id];
        }

        std::vector<int> neighbor_indices;
        float search_radius_squared = param.radius_ * param.radius_;
        for (int vkernel = -kernel_radius; vkernel <= kernel_radius; vkernel++) {
          for (int ukernel = -kernel_radius; ukernel <= kernel_radius; ukernel++) {
            int y = v + vkernel;
            int x = u + ukernel;

            if (x > 0 && y > 0 && x < (int)cloud->width && y < (int)cloud->height) {
              const PointT &pt1 = cloud->at(x, y);
              if (pcl::isFinite(pt1)) {
                if (param.adaptive_) {
                  const float center_dist_squared = vkernel * vkernel + ukernel * ukernel;
                  // const float val = param_.kappa_ * sqrt(center_dist_squared) * pt1.z + param_.d_;
                  // new_sqr_radius = val * val;
                  search_radius_squared = param.kappa_ * param.kappa_ * center_dist_squared * pt1.z * pt1.z + param.d_;
                }

                if ((p.getVector4fMap() - pt1.getVector4fMap()).squaredNorm() < search_radius_squared)
                  neighbor_indices.push_back(y * cloud->width + x);
              }
            }
          }
        }

        n = computeNormal(*cloud, neighbor_indices);
        pcl::flipNormalTowardsViewpoint(p, 0.f, 0.f, 0.f, n.normal_x, n.normal_y, n.normal_z);
      }
    } break;
  }
  return normals;
}

#define PCL_INSTANTIATE_computeNormals(T)                                   \
  template pcl::PointCloud<pcl::Normal>::Ptr computeNormals<T>( \
      const typename pcl::PointCloud<T>::ConstPtr, const NormalEstimatorParameter &, const pcl::IndicesConstPtr);
PCL_INSTANTIATE(computeNormals, (pcl::PointXYZ)(pcl::PointXYZRGB)(pcl::PointNormal)(pcl::PointXYZRGBNormal))
/// TODO Use V4R_PCL_XYZ_POINT_TYPES - requires to fix circular dependency with common module
}  // namespace v4r
