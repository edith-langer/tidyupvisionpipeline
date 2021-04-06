#include <glog/logging.h>
#include <v4r/apps/CloudSegmenter.h>
#include <v4r/common/miscellaneous.h>
#include <v4r/common/point_types.h>
#include <v4r/segmentation/plane_utils.h>
// #include <v4r/segmentation/segmentation_utils.h>

#include <pcl/common/time.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/impl/instantiate.hpp>

#include <boost/format.hpp>

namespace po = boost::program_options;

namespace v4r {
namespace apps {

void CloudSegmenterParameter::init(boost::program_options::options_description &desc, const std::string &section_name) {
  desc.add_options()((section_name + ".chop_z,z").c_str(), po::value<float>(&chop_z_)->default_value(chop_z_),
                     "cut-off threshold in meter");
  desc.add_options()((section_name + ".min_plane_inliers").c_str(),
                     po::value<size_t>(&min_plane_inliers_)->default_value(min_plane_inliers_),
                     "minimum number of inlier points for a plane to be valid");
  desc.add_options()((section_name + ".min_plane_diameter").c_str(),
                     po::value<float>(&min_plane_diameter_)->default_value(min_plane_diameter_),
                     "minimum plane diameter in meter for a plane to be valid");
  desc.add_options()((section_name + ".skip_segmentation").c_str(),
                     po::value<bool>(&skip_segmentation_)->default_value(skip_segmentation_),
                     " if true, skips segmentation");
  desc.add_options()((section_name + ".remove_planes").c_str(),
                     po::value<bool>(&remove_planes_)->default_value(remove_planes_),
                     "if true, removes plane from input cloud only. If false, removes plane and "
                     "everything below it (i.e. further away from the camera)");
  desc.add_options()((section_name + ".remove_selected_plane").c_str(),
                     po::value<bool>(&remove_selected_plane_)->default_value(remove_selected_plane_),
                     "if true, removes the selected plane (either dominant or the one parallel and higher)");
  desc.add_options()(
      (section_name + ".remove_points_below_selected_plane").c_str(),
      po::value<bool>(&remove_points_below_selected_plane_)->default_value(remove_points_below_selected_plane_),
      "if true, removes only the plane with the largest number of plane inliers");
  desc.add_options()(
      (section_name + ".use_highest_plane").c_str(),
      po::value<bool>(&use_highest_plane_)->default_value(use_highest_plane_),
      "if true, removes all points which are not above the highest plane parallel to the dominant plane");
  desc.add_options()(
      (section_name + ".cosinus_angle_for_planes_to_be_parallel").c_str(),
      po::value<float>(&cosinus_angle_for_planes_to_be_parallel_)
          ->default_value(cosinus_angle_for_planes_to_be_parallel_),
      "the minimum cosinus angle of the surface normals of two planes such that the two planes are considered "
      "parallel (only used if check for higher plane is enabled)");
  desc.add_options()((section_name + ".min_distance_to_plane").c_str(),
                     po::value<float>(&min_distance_to_plane_)->default_value(min_distance_to_plane_),
                     "minimum distance in meter a point needs to have to be considered above");
  desc.add_options()((section_name + ".segmentation_method").c_str(),
                     po::value<SegmentationType>(&segmentation_method_)->default_value(segmentation_method_),
                     "segmentation method");
  desc.add_options()((section_name + ".plane_extraction_method").c_str(),
                     po::value<PlaneExtractionType>(&plane_extraction_method_)->default_value(plane_extraction_method_),
                     "plane extraction method");
  plane_extractor_parameter_.init(desc, section_name + ".plane_extractor");
  normal_computation_parameter_.init(desc, section_name + ".normal_estimator");
  segmentation_parameter_.init(desc, section_name + ".segmenter");
}

template <typename PointT>
void CloudSegmenter<PointT>::initialize() {
  plane_extractor_ =
      v4r::initPlaneExtractor<PointT>(param_.plane_extraction_method_, param_.plane_extractor_parameter_);

  if (!param_.skip_segmentation_)
    segmenter_ = v4r::initSegmenter<PointT>(param_.segmentation_method_, param_.segmentation_parameter_);
}

template <typename PointT>
void CloudSegmenter<PointT>::segment(const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
                                     const boost::optional<Eigen::Matrix4f> &transform_to_world) {
  processed_cloud_.reset(new pcl::PointCloud<PointT>(*cloud));

  if (!normals_ && ((segmenter_ && segmenter_->getRequiresNormals()) ||
                    (plane_extractor_ && plane_extractor_->getRequiresNormals()))) {
    normals_ = computeNormals<PointT>(cloud, param_.normal_computation_parameter_);
  }

  for (PointT &p : processed_cloud_->points) {
    if (pcl::isFinite(p) && p.z > param_.chop_z_)
      p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
  }

  if (!param_.skip_plane_extraction_) {
    pcl::StopWatch t;
    plane_extractor_->setInputCloud(processed_cloud_);
    plane_extractor_->setNormalsCloud(normals_);
    if (transform_to_world) {
      const Eigen::Vector4f search_axis_world = Eigen::Vector4f::UnitZ();
      const Eigen::Vector3f search_axis_camera = (transform_to_world.get().transpose() * search_axis_world).head(3);
      plane_extractor_->compute(search_axis_camera);
    } else {
      plane_extractor_->compute();
    }
    planes_ = plane_extractor_->getPlanes();
    plane_inliers_ = plane_extractor_->getPlaneInliers();

    // remove large planes
    size_t kept = 0;
    for (size_t plane_id = 0; plane_id < planes_.size(); plane_id++) {
      pcl::ConvexHull<PointT> convex_hull;
      typename pcl::PointCloud<PointT>::Ptr plane_cloud(new pcl::PointCloud<PointT>);  // Note: unfortunately PCL convex
                                                                                       // hull computation crashes if
                                                                                       // indices are used directly
      pcl::copyPointCloud(*processed_cloud_, plane_inliers_[plane_id], *plane_cloud);
      convex_hull.setInputCloud(plane_cloud);
      pcl::PointCloud<PointT> convex_hull_pts;
      convex_hull.reconstruct(convex_hull_pts);

      if (computePointcloudDiameter(convex_hull_pts) > param_.min_plane_diameter_) {
        plane_inliers_[kept] = plane_inliers_[plane_id];
        planes_[kept] = planes_[plane_id];
        kept++;
      }
    }
    plane_inliers_.resize(kept);
    planes_.resize(kept);

    CHECK(plane_inliers_.size() == planes_.size());

    // discard planes without sufficient plane inliers
    kept = 0;
    for (size_t plane_id = 0; plane_id < planes_.size(); plane_id++) {
      if (plane_inliers_[plane_id].size() > param_.min_plane_inliers_) {
        plane_inliers_[kept] = plane_inliers_[plane_id];
        planes_[kept] = planes_[plane_id];
        kept++;
      }
    }
    plane_inliers_.resize(kept);
    planes_.resize(kept);

    if (planes_.empty()) {
      LOG(WARNING) << "Could not extract any plane with the chosen parameters. Segmenting the whole input cloud!";

      if (transform_to_world) {
        const Eigen::Vector4f search_axis_world = Eigen::Vector4f::UnitZ();
        selected_plane_ = transform_to_world.get().transpose() * search_axis_world;
        LOG(INFO) << "Setting table plane to " << search_axis_world.transpose() << " in world reference frame.";
      }
    } else {
      // get dominant plane id
      size_t dominant_plane_id = 0;
      size_t max_inliers = 0;
      for (size_t plane_id = 0; plane_id < plane_inliers_.size(); plane_id++) {
        if (plane_inliers_[plane_id].size() > max_inliers) {
          max_inliers = plane_inliers_[plane_id].size();
          dominant_plane_id = plane_id;
        }
      }
      size_t selected_plane_id = dominant_plane_id;
      // check if there is a plane higher than the dominant plane which is parallel to it
      if (param_.use_highest_plane_) {
        for (size_t plane_id = 0; plane_id < plane_inliers_.size(); plane_id++) {
          if (selected_plane_id == plane_id)
            continue;

          const Eigen::Vector4f &current_highest_plane = planes_[selected_plane_id];
          const Eigen::Vector4f &plane_tmp = planes_[plane_id];

          if (CosAngleBetweenPlanes(plane_tmp, current_highest_plane) >
              param_.cosinus_angle_for_planes_to_be_parallel_)  // check if parallel
          {
            // check if higher
            if (plane_tmp(3) / plane_tmp.head(3).norm() <
                current_highest_plane(3) / current_highest_plane.head(3).norm()) {
              selected_plane_id = plane_id;
            }
          }
        }
      }

      selected_plane_ = planes_[selected_plane_id];

      // now filter
      if (param_.remove_planes_ || param_.remove_selected_plane_) {
        for (size_t plane_id = 0; plane_id < plane_inliers_.size(); plane_id++) {
          if (plane_id != selected_plane_id && !param_.remove_planes_)
            continue;

          for (int idx : plane_inliers_[plane_id]) {
            PointT &p = processed_cloud_->points[idx];
            p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
          }
        }
      }

      if (param_.remove_points_below_selected_plane_) {
        std::vector<int> above_plane_indices =
            v4r::get_above_plane_inliers(*processed_cloud_, planes_[selected_plane_id], param_.min_distance_to_plane_);
        boost::dynamic_bitset<> above_plane_mask =
            v4r::createMaskFromIndices(above_plane_indices, processed_cloud_->points.size());

        for (size_t i = 0; i < processed_cloud_->points.size(); i++)  // keep organized
        {
          if (!above_plane_mask[i]) {
            PointT &p = processed_cloud_->points[i];
            p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
          }
        }
      }
    }
    VLOG(1) << "Plane extraction took " << t.getTime() << " ms.";
  }

  if (!param_.skip_segmentation_) {
    pcl::StopWatch t;
    segmenter_->setInputCloud(processed_cloud_);
    segmenter_->setNormalsCloud(normals_);  // since the cloud was kept organized, we can use the original normal cloud
    segmenter_->segment();
    segmenter_->getSegmentIndices(found_clusters_);
    VLOG(1) << "Segmentation took " << t.getTime() << " ms.";
  }
}

#define PCL_INSTANTIATE_CloudSegmenter(T) template class CloudSegmenter<T>;
PCL_INSTANTIATE(CloudSegmenter, V4R_PCL_XYZ_POINT_TYPES)
}  // namespace apps
}  // namespace v4r
