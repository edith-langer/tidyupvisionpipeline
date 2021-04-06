#include <v4r/common/miscellaneous.h>
#include <v4r/common/time.h>
//#include <v4r/config.h>
#include <v4r/geometry/normals.h>
#include <v4r/io/eigen.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/model.h>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/surface/convex_hull.h>

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

constexpr const auto kSymmetryXyz = "symmetry_xyz";
constexpr const auto kRotationalInvariance = "rotational_invariance_xyz";

namespace v4r {

ModelProperties::ModelProperties(const bf::path &metadata_file) {
  if (bf::exists(metadata_file)) {
#if HAVE_YAML_CPP
    YAML::Node config = YAML::LoadFile(metadata_file.string());
    if (config[kSymmetryXyz]) {
      const auto symmetry_xyz = config[kSymmetryXyz].as<std::string>();
      std::stringstream iss(symmetry_xyz);
      bool b1, b2, b3;
      iss >> b1 >> b2 >> b3;
      symmetry_xyz_ = {b1, b2, b3};
    }
    if (config[kRotationalInvariance]) {
      const auto symmetry_xyz = config[kRotationalInvariance].as<std::string>();
      std::stringstream iss(symmetry_xyz);
      bool b1, b2, b3;
      iss >> b1 >> b2 >> b3;
      rotational_invariance_xyz_ = {b1, b2, b3};
    }
#else
    LOG(ERROR) << "Yaml-Cpp library is not installed. Therefore cannot read " << metadata_file << "!";
#endif
  } else {
    LOG(WARNING) << "Metadata yaml file " << metadata_file
                 << " does not exist! Will use default object model properties.";
  }
}

template <typename PointT>
void Model<PointT>::initialize(const bf::path &model_filename, const NormalEstimatorParameter &ne_param) {
  typename pcl::PointCloud<PointTWithNormal>::Ptr all_assembled(new pcl::PointCloud<PointTWithNormal>);
  if (!io::existsFile(model_filename) || pcl::io::loadPCDFile(model_filename.string(), *all_assembled) == -1) {
    v4r::ScopeTime t("Creating 3D model");
    typename pcl::PointCloud<PointT>::Ptr accumulated_cloud(
        new pcl::PointCloud<PointT>);  ///< 3D point cloud taking into account all training views
    pcl::PointCloud<pcl::Normal>::Ptr accumulated_normals(new pcl::PointCloud<pcl::Normal>);  /// corresponding normals
    /// to the 3D point cloud
    /// taking into account
    /// all training views

    /// TODO use noise model and voxel grid to merge point clouds. For now just accumulate all points.
    for (const typename TrainingView<PointT>::ConstPtr &v : views_) {
      typename pcl::PointCloud<PointT>::ConstPtr cloud;
      pcl::PointCloud<pcl::Normal>::ConstPtr normals;
      std::vector<int> indices;
      Eigen::Matrix4f tf;
      typename pcl::PointCloud<PointT>::Ptr obj_cloud_tmp(new pcl::PointCloud<PointT>);
      pcl::PointCloud<pcl::Normal>::Ptr obj_normals_tmp(new pcl::PointCloud<pcl::Normal>);

      if (v->cloud_) {
        cloud = v->cloud_;
        indices = v->indices_;
        pcl::copyPointCloud(*v->cloud_, indices, *obj_cloud_tmp);
      } else {
        typename pcl::PointCloud<PointT>::Ptr cloud_tmp(new pcl::PointCloud<PointT>);
        pcl::io::loadPCDFile(v->filename_.string(), *cloud_tmp);
        cloud = cloud_tmp;
      }

      if (v->normals_)
        normals = v->normals_;
      else {
        normals = computeNormals<PointT>(cloud, ne_param);
      }

      if (!v->indices_.empty())
        indices = v->indices_;
      else {
        std::ifstream mi_f(v->indices_filename_.string());
        int idx;
        while (mi_f >> idx)
          indices.push_back(idx);
        mi_f.close();
      }

      try {
        tf = io::readMatrixFromFile(v->pose_filename_);
      } catch (const std::runtime_error &e) {
        tf = Eigen::Matrix4f::Identity();
      }

      if (indices.empty()) {
        pcl::copyPointCloud(*cloud, *obj_cloud_tmp);
        pcl::copyPointCloud(*normals, *obj_normals_tmp);
      } else {
        pcl::copyPointCloud(*cloud, indices, *obj_cloud_tmp);
        pcl::copyPointCloud(*normals, indices, *obj_normals_tmp);
      }
      pcl::transformPointCloud(*obj_cloud_tmp, *obj_cloud_tmp, tf);
      transformNormals(*obj_normals_tmp, *obj_normals_tmp, tf);

      *accumulated_cloud += *obj_cloud_tmp;
      *accumulated_normals += *obj_normals_tmp;
    }

    if (accumulated_cloud->points.size() != accumulated_normals->points.size())
      std::cerr << "Point cloud and normals point cloud of model created by accumulating all points from training does "
                   "not have the same size! This can lead to undefined behaviour!"
                << std::endl;

    pcl::concatenateFields(*accumulated_cloud, *accumulated_normals, *all_assembled);

    if (!model_filename.string().empty()) {
      io::createDirForFileIfNotExist(model_filename);
      pcl::io::savePCDFileBinaryCompressed(model_filename.string(), *all_assembled);
    }
  }

  /// 3D model exists
  else {
      pcl::PointCloud<pcl::Normal>::Ptr normals = computeNormals<PointTWithNormal>(all_assembled, ne_param);

      pcl::PCLPointCloud2* header = new pcl::PCLPointCloud2;
      pcl::PCDReader reader;
      reader.readHeader(model_filename.string(), *header);
      if(std::find_if(header->fields.begin(), header->fields.end(), [](const pcl::PCLPointField & field) {
                      return field.name.find("normal") != std::string::npos;
  }) != header->fields.end()) {
          //model has already normals, refine them with the computed normals
          for (size_t p = 0; p < all_assembled->size(); p++) {
              const auto normal_pre = all_assembled->at(p).getNormalVector4fMap();
              const auto normal_comp = normals->at(p).getNormalVector4fMap();
              float normals_dotp = normal_comp.dot(normal_pre);
              if (normals_dotp < 0) { //negative dot product = angle between the two normals is between 90 and 270 deg
                  normals->at(p).normal_x *= -1;
                  normals->at(p).normal_y *= -1;
                  normals->at(p).normal_z *= -1;
              }
          }
      }
      pcl::concatenateFields(*all_assembled, *normals, *all_assembled);
  }

  all_assembled_ = all_assembled;
  pcl::getMinMax3D(*all_assembled_, minPoint_, maxPoint_);
  pcl::ConvexHull<PointTWithNormal> convex_hull;
  convex_hull.setInputCloud(all_assembled_);
  convex_hull_points_.reset(new pcl::PointCloud<PointTWithNormal>);
  convex_hull.reconstruct(*convex_hull_points_);
  diameter_ = computePointcloudDiameter(*convex_hull_points_);
  cluster_props_.reset(new Cluster(*all_assembled_));
}

template class Model<pcl::PointXYZ>;
template class Model<pcl::PointXYZRGB>;

}  // namespace v4r
