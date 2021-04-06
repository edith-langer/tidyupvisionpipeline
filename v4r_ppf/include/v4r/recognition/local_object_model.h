

#include <pcl/common/common.h>

/**
 * @brief The LocalObjectModel class stores information about the object model related to local feature extraction
 */
class  LocalObjectModel {
 public:
  pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints_;  ///< all extracted keypoints of the object model
  pcl::PointCloud<pcl::Normal>::Ptr
      kp_normals_;  ///< normals associated to each extracted keypoints of the object model

  LocalObjectModel() {
    keypoints_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    kp_normals_.reset(new pcl::PointCloud<pcl::Normal>);
  }

  typedef std::shared_ptr<LocalObjectModel> Ptr;
  typedef std::shared_ptr<LocalObjectModel const> ConstPtr;
};