#include <pcl/common/transforms.h>
#include <pcl/features/organized_edge_detection.h>
#include <pcl/octree/octree_pointcloud_pointvector.h>
#include <pcl/pcl_config.h>
#include <v4r/common/noise_models.h>
#include <v4r/geometry/normals.h>
#include <v4r/registration/noise_model_based_cloud_integration.h>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <pcl/octree/impl/octree_iterator.hpp>

#include <glog/logging.h>
#include <omp.h>

namespace po = boost::program_options;

namespace v4r {

void NMBasedCloudIntegrationParameter::init(boost::program_options::options_description &desc,
                                            const std::string &section_name) {
  desc.add_options()((section_name + ".resolution").c_str(),
                     po::value<float>(&octree_resolution_)->default_value(octree_resolution_),
                     "resolution of the octree of the big point cloud");
  desc.add_options()((section_name + ".min_points_per_voxel").c_str(),
                     po::value<size_t>(&min_points_per_voxel_)->default_value(min_points_per_voxel_),
                     "the minimum number of points in a leaf of the octree of the big cloud->");
  desc.add_options()((section_name + ".min_px_distance_to_depth_discontinuity").c_str(),
                     po::value<float>(&min_px_distance_to_depth_discontinuity_)
                         ->default_value(min_px_distance_to_depth_discontinuity_),
                     "points of the input cloud within this distance (in pixel) to its closest depth discontinuity "
                     "pixel will be removed");
  desc.add_options()((section_name + ".average").c_str(), po::value<bool>(&average_)->default_value(average_),
                     "if true, takes the average color (for each color componenent) and normal within all the points "
                     "in the leaf of the octree. Otherwise, it takes the point within the octree with the best noise "
                     "weight");
  desc.add_options()((section_name + ".viewpoint_surface_orienation_dotp_thresh").c_str(),
                     po::value<float>(&viewpoint_surface_orienation_dotp_thresh_)
                         ->default_value(viewpoint_surface_orienation_dotp_thresh_),
                     "threshold for the inner product of viewpoint and surface normal orientation to have more "
                     "importance than the weight variable");
  desc.add_options()(
      (section_name + ".px_distance_to_depth_discontinuity_thresh").c_str(),
      po::value<float>(&px_distance_to_depth_discontinuity_thresh_)
          ->default_value(px_distance_to_depth_discontinuity_thresh_),
      "threshold up to what point the distance to depth discontinuity is more important than other metrics");
  desc.add_options()((section_name + ".resolution_adaptive_min_points").c_str(),
                     po::value<bool>(&resolution_adaptive_min_points_)->default_value(resolution_adaptive_min_points_),
                     "if true, sets the requirements for the minimum number of points to a certain percentage of the "
                     "maximum number of points within a voxel");
  desc.add_options()(
      (section_name + ".adaptive_min_points_percentage_thresh").c_str(),
      po::value<float>(&adaptive_min_points_percentage_thresh_)->default_value(adaptive_min_points_percentage_thresh_),
      "if resolution_adaptive_min_points_ is set to true, this parameter will set the minimum number of points needed "
      "within each voxel as percentage of the maximum number of points within each voxel");
  desc.add_options()((section_name + ".use_nguyen").c_str(), po::value<bool>(&use_nguyen_)->default_value(use_nguyen_),
                     "if true, uses Nguyens noise model (Nguyen et al., 3DIMPVT 2012.)");
}

template <typename PointT>
void NMBasedCloudIntegration<PointT>::addView(const typename pcl::PointCloud<PointT>::ConstPtr &cloud,
                                              const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
                                              const Eigen::Matrix4f &transform_to_global_reference_frame,
                                              const boost::optional<std::vector<int>> &indices) {
  // pre-allocate memory
  size_t old_size = big_cloud_info_.size();
  size_t max_new_pts = indices ? old_size + indices.get().size() : cloud->size();
  big_cloud_info_.resize(old_size + max_new_pts);
  cv::Mat img_boundary_distance;
  std::vector<std::vector<float>> pt_properties;

  // compute distance (in pixels) to edge for each pixel
  if (param_.min_px_distance_to_depth_discontinuity_ > 0) {
    // compute depth discontinuity edges
    pcl::OrganizedEdgeBase<PointT, pcl::Label> oed;
    oed.setDepthDisconThreshold(0.05f);  // at 1m, adapted linearly with depth
    oed.setMaxSearchNeighbors(100);
    oed.setEdgeType(pcl::OrganizedEdgeBase<PointT, pcl::Label>::EDGELABEL_OCCLUDING |
                    pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_OCCLUDED |
                    pcl::OrganizedEdgeBase<pcl::PointXYZRGB, pcl::Label>::EDGELABEL_NAN_BOUNDARY);
    oed.setInputCloud(cloud);

    pcl::PointCloud<pcl::Label>::Ptr labels(new pcl::PointCloud<pcl::Label>);
    std::vector<pcl::PointIndices> edge_indices;
    oed.compute(*labels, edge_indices);

    cv::Mat_<uchar> pixel_is_edge(cloud->height, cloud->width);  // depth edges
    pixel_is_edge.setTo(255);

    for (size_t j = 0; j < edge_indices.size(); j++) {
      for (int edge_px_id : edge_indices[j].indices) {
        int row = edge_px_id / cloud->width;
        int col = edge_px_id % cloud->width;
        pixel_is_edge.at<uchar>(row, col) = 0;
      }
    }
    cv::distanceTransform(pixel_is_edge, img_boundary_distance, CV_DIST_L2, 5);
  }

  if (param_.use_nguyen_) {
    pt_properties = NguyenNoiseModel<PointT>::compute(cloud, normals, cam_->fx);
  }

  pcl::PointCloud<PointT> cloud_aligned;
  pcl::PointCloud<pcl::Normal> normals_aligned;
  pcl::transformPointCloud(*cloud, cloud_aligned, transform_to_global_reference_frame);
  v4r::transformNormals(*normals, normals_aligned, transform_to_global_reference_frame);

  size_t kept_new_pts = 0;
  std::vector<int> fake_indices;
  if (indices) {
    fake_indices = indices.get();
  } else {
    fake_indices.resize(cloud_aligned.size());
    std::iota(std::begin(fake_indices), std::end(fake_indices), 0);
  }

  for (int idx : fake_indices) {
    const auto &p_orig = cloud->points[idx];
    const auto &n_orig = normals->points[idx];
    const auto &p_aligned = cloud_aligned.points[idx];
    const auto &n_aligned = normals_aligned.points[idx];

    if (!pcl::isFinite(p_aligned) || !pcl::isFinite(n_aligned))
      continue;

    auto &pt = big_cloud_info_[old_size + kept_new_pts];
    pt.pt_ = p_aligned;
    pt.normal_ = n_aligned;
    pt.dotp_ = p_orig.getVector3fMap().normalized().dot(n_orig.getNormalVector3fMap());

    if (!img_boundary_distance.empty()) {
      int row = idx / cloud->width;
      int col = idx % cloud->width;
      pt.distance_to_depth_discontinuity_ = img_boundary_distance.at<float>(row, col);
    }

    if (param_.use_nguyen_) {
      const auto sigma_lateral_ = pt_properties[idx][0];
      const auto sigma_axial_ = pt_properties[idx][1];
      const Eigen::DiagonalMatrix<float, 3> sigma(sigma_lateral_, sigma_lateral_, sigma_axial_);
      const Eigen::Matrix4f &tf = transform_to_global_reference_frame;
      Eigen::Matrix3f rotation = tf.block<3, 3>(0, 0);  // or inverse?
      Eigen::Matrix3f sigma_aligned = rotation * sigma * rotation.transpose();
      double det = sigma_aligned.determinant();

      //      if( std::isfinite(det) && det>0)
      //          pt.probability = 1 / sqrt(2 * M_PI * det);
      //      else
      //          pt.probability = std::numeric_limits<float>::min();

      if (std::isfinite(det) && det > 0.)
        pt.weight_ = det;
      else
        pt.weight_ = std::numeric_limits<float>::max();
    } else {
      pt.weight_ = p_orig.getVector3fMap().norm();
    }

    kept_new_pts++;
  }
  big_cloud_info_.resize(old_size + kept_new_pts);
}

template <typename PointT>
void NMBasedCloudIntegration<PointT>::compute(typename pcl::PointCloud<PointT>::Ptr &output) {
  pcl::octree::OctreePointCloudPointVector<PointT> octree(param_.octree_resolution_);
  typename pcl::PointCloud<PointT>::Ptr big_cloud(new pcl::PointCloud<PointT>());
  big_cloud->resize(big_cloud_info_.size());
  for (size_t i = 0; i < big_cloud_info_.size(); i++)
    big_cloud->points[i] = big_cloud_info_[i].pt_;
  octree.setInputCloud(big_cloud);
  octree.addPointsFromInputCloud();

  typename pcl::octree::OctreePointCloudPointVector<PointT>::LeafNodeIterator leaf_it;

#if PCL_VERSION_COMPARE(>=, 1, 9, 0)
  const typename pcl::octree::OctreePointCloudPointVector<PointT>::LeafNodeIterator it2_end = octree.leaf_depth_end();
#else
  const typename pcl::octree::OctreePointCloudPointVector<PointT>::LeafNodeIterator it2_end = octree.leaf_end();
#endif

  size_t kept = 0;
  size_t total_used = 0;
  size_t min_points_per_voxel = param_.min_points_per_voxel_;

  if (param_.resolution_adaptive_min_points_) {
    size_t max_pts_per_voxel = 0;
#if PCL_VERSION_COMPARE(>=, 1, 9, 0)
    for (leaf_it = octree.leaf_depth_begin(); leaf_it != it2_end; ++leaf_it) {
#else
    for (leaf_it = octree.leaf_begin(); leaf_it != it2_end; ++leaf_it) {
#endif
      pcl::octree::OctreeContainerPointIndices &container = leaf_it.getLeafContainer();

      // add points from leaf node to indexVector
      std::vector<int> indexVector;
      container.getPointIndices(indexVector);

      if (indexVector.size() > max_pts_per_voxel)
        max_pts_per_voxel = indexVector.size();
    }
    min_points_per_voxel =
        std::max<size_t>(param_.min_points_per_voxel_,
                         static_cast<size_t>(param_.adaptive_min_points_percentage_thresh_ * max_pts_per_voxel));
    LOG(INFO) << "Adaptive filtering enable with a threshold computed at " << min_points_per_voxel
              << " points given a maximum number of " << max_pts_per_voxel << " and an adaptive threshold of "
              << param_.adaptive_min_points_percentage_thresh_;
  }

  std::vector<PointInfo> filtered_cloud_info(big_cloud_info_.size());

#if PCL_VERSION_COMPARE(>=, 1, 9, 0)
  for (leaf_it = octree.leaf_depth_begin(); leaf_it != it2_end; ++leaf_it) {
#else
  for (leaf_it = octree.leaf_begin(); leaf_it != it2_end; ++leaf_it) {
#endif
    pcl::octree::OctreeContainerPointIndices &container = leaf_it.getLeafContainer();

    // add points from leaf node to indexVector
    std::vector<int> indexVector;
    container.getPointIndices(indexVector);

    if (indexVector.empty())
      continue;

    std::vector<PointInfo> voxel_pts(indexVector.size());

    for (size_t k = 0; k < indexVector.size(); k++)
      voxel_pts[k] = big_cloud_info_[indexVector[k]];

    size_t num_good_pts = std::count_if(voxel_pts.begin(), voxel_pts.end(), [this](const PointInfo &p) {
      return p.distance_to_depth_discontinuity_ > this->param_.min_px_distance_to_depth_discontinuity_;
    });

    if (num_good_pts < min_points_per_voxel)
      continue;

    PointInfo p;

    if (param_.average_) {
      for (const PointInfo &pt_tmp : voxel_pts) {
        if (pt_tmp.distance_to_depth_discontinuity_ > param_.min_px_distance_to_depth_discontinuity_) {
          p.moving_average(pt_tmp);
        }
      }
      total_used += num_good_pts;
    } else {
      // now comes the actual magic. We only return the point with minimum weight within this voxel. Except the viewray
      // to surface configuration is so bad that we rather return a point whose surface is more facing the camera.
      std::sort(voxel_pts.begin(), voxel_pts.end(), [this](const PointInfo &a, const PointInfo &b) {
        if ((a.distance_to_depth_discontinuity_ < param_.px_distance_to_depth_discontinuity_thresh_ ||
             b.distance_to_depth_discontinuity_ < param_.px_distance_to_depth_discontinuity_thresh_) &&
            (static_cast<int>(a.distance_to_depth_discontinuity_) !=
             static_cast<int>(b.distance_to_depth_discontinuity_)))
          return a.distance_to_depth_discontinuity_ > b.distance_to_depth_discontinuity_;

        // we take a minus here because the viewpoint and the surface normal will have a negative inner product
        if (a.dotp_ > -param_.viewpoint_surface_orienation_dotp_thresh_ ||
            b.dotp_ > -param_.viewpoint_surface_orienation_dotp_thresh_)
          return a.dotp_ < b.dotp_;

        return a.weight_ < b.weight_;
      });

      const auto it = std::find_if(voxel_pts.begin(), voxel_pts.end(), [this](const auto &pt) -> bool {
        return pt.distance_to_depth_discontinuity_ > param_.min_px_distance_to_depth_discontinuity_;
      });

      if (it != voxel_pts.end())
        p = *it;

      total_used++;
    }
    filtered_cloud_info[kept++] = p;
  }
  filtered_cloud_info.resize(kept);

  LOG(INFO) << "Number of points in final noise model based integrated cloud: " << kept << " used: " << total_used;

  if (!output)
    output.reset(new pcl::PointCloud<PointT>);

  if (!output_normals_)
    output_normals_.reset(new pcl::PointCloud<pcl::Normal>);

  output->resize(kept);
  output_normals_->resize(kept);
  output->is_dense = output_normals_->is_dense = true;

  for (size_t i = 0; i < filtered_cloud_info.size(); i++) {
    output_normals_->points[i] = filtered_cloud_info[i].normal_;
    output->points[i] = filtered_cloud_info[i].pt_;
  }
  cleanUp();
}

template class NMBasedCloudIntegration<pcl::PointXYZRGB>;
}  // namespace v4r
