#ifndef PLANE_OBJECT_EXTRACTION_H
#define PLANE_OBJECT_EXTRACTION_H

#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <limits>
#include <algorithm>

#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/common/centroid.h>
#include <pcl/filters/crop_hull.h>

#include "region_growing.h"
#include "object_matching.h"
#include "mathhelpers.h"

typedef pcl::PointXYZRGBNormal PointNormal;
typedef pcl::PointXYZRGB PointRGB;
typedef pcl::PointXYZ PointXYZ;


class ExtractObjectsFromPlanes {
public:
    ExtractObjectsFromPlanes(pcl::PointCloud<PointNormal>::Ptr cloud, Eigen::Vector4f main_plane_coeffs, pcl::PointCloud<pcl::PointXYZ>::Ptr convex_hull_pts,
                             std::string output_path="");
    ~ExtractObjectsFromPlanes();

    void setResultPath(std::string path);

    std::vector<PlaneWithObjInd> computeObjectsOnPlanes(pcl::PointCloud<PointNormal>::Ptr checked_plane_points_cloud); //this method returns potential object (points) that correspond to new or disappeard objects supported by a plane


private:
    pcl::PointCloud<PointNormal>::Ptr orig_cloud_;
    Eigen::Vector4f main_plane_coeffs_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr convex_hull_pts_;
    std::string result_path_;
    std::vector<PlaneStruct> hor_planes_;

    std::vector<PlaneWithObjInd> extractObjectInd(pcl::PointCloud<PointNormal>::Ptr checked_plane_points_cloud);
    void filter_flying_objects(pcl::PointCloud<PointNormal>::Ptr orig_cloud_, pcl::PointIndices::Ptr ind, pcl::PointCloud<PointNormal>::Ptr plane);
    pcl::PointIndices::Ptr findCorrespondingPlane(pcl::PointCloud<PointNormal>::Ptr cloud, std::vector<pcl::PointIndices> clusters);
    void shrinkConvexHull(pcl::PointCloud<PointNormal>::Ptr hull_cloud, float distance);
    void filter_planar_objects(pcl::PointCloud<PointNormal>::Ptr cloud, pcl::PointIndices::Ptr ind);
    bool intersectCHWithPlane(pcl::PointCloud<PointXYZ>::ConstPtr cloud_hull_orig, pcl::PointCloud<PointNormal>::ConstPtr plane_cloud);
};

#endif //PLANE_OBJECT_EXTRACTION_H
