#ifndef CHANGE_DETECTION_H
#define CHANGE_DETECTION_H

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <chrono>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/impl/search.hpp>

#include <boost/filesystem.hpp>

#include <v4r/geometry/normals.h>

#include "scene_differencing_points.h"
#include "local_object_verification.h"
#include "object_visualization.h"
#include "region_growing.h"
#include "color_histogram.h"
#include "detected_object.h"
#include "object_matching.h"
#include "plane_object_extraction.h"
#include "color_histogram.h"

typedef pcl::PointXYZRGBNormal PointNormal;
typedef pcl::PointXYZRGBL PointLabel;


class ChangeDetection
{
public:
    ChangeDetection(std::string ppf_config_path) : ppf_config_path_(ppf_config_path) {}

    void init(pcl::PointCloud<PointNormal>::Ptr ref_cloud, pcl::PointCloud<PointNormal>::Ptr curr_cloud,
              Eigen::Vector4f ref_plane_coeffs, Eigen::Vector4f curr_plane_coeffs,
              pcl::PointCloud<pcl::PointXYZ>::Ptr ref_convex_hull_pts, pcl::PointCloud<pcl::PointXYZ>::Ptr curr_convex_hull_pts,
              std::string output_path = "");

    void setOutputPath (std::string output_path) {
        output_path_ = output_path;
    }

    void setRefCloud(pcl::PointCloud<PointNormal>::Ptr cloud) {
        this->ref_cloud_ = cloud;
    }

    void setCurrCloud(pcl::PointCloud<PointNormal>::Ptr cloud) {
        this->curr_cloud_ = cloud;
    }

    void compute(std::vector<DetectedObject> &ref_result, std::vector<DetectedObject> &curr_result);

private:
    //std::string object_store_path_; //the model objects and their ppf model get stored here --> for now we store everything in output path
    std::string ppf_config_path_; //the config file that stores the parrameters for PPF
    std::string output_path_; //all debugging things will get stored there (+model objects and their ppf model)

    std::vector<PlaneWithObjInd> potential_objects_;
    pcl::PointCloud<PointNormal>::Ptr ref_cloud_;
    pcl::PointCloud<PointNormal>::Ptr curr_cloud_;

    Eigen::Vector4f ref_plane_coeffs_;
    Eigen::Vector4f curr_plane_coeffs_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr curr_convex_hull_pts_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr ref_convex_hull_pts_;

    bool do_LV_before_matching = false;


    void refineNormals(pcl::PointCloud<PointNormal>::Ptr object_cloud);
    std::vector<pcl::PointIndices> removeClusteroutliersBySize(pcl::PointCloud<PointNormal>::Ptr cloud, float cluster_thr, int min_cluster_size=15, int max_cluster_size=std::numeric_limits<int>::max());
    pcl::PointCloud<PointNormal>::Ptr downsampleCloud(pcl::PointCloud<PointNormal>::Ptr input, double leafSize);
    void upsampleObjectsAndPlanes(pcl::PointCloud<PointNormal>::Ptr orig_cloud, pcl::PointCloud<PointNormal>::Ptr ds_cloud,
                                  std::vector<PlaneWithObjInd> &objects, double leaf_size, std::string res_path);
    std::tuple<pcl::PointCloud<PointNormal>::Ptr, std::vector<int> > upsampleObjects(pcl::octree::OctreePointCloudSearch<PointNormal>::Ptr octree, pcl::PointCloud<PointNormal>::Ptr orig_input_cloud,
                                                                                     pcl::PointCloud<PointNormal>::Ptr objects_ds_cloud, std::string output_path, int counter);
    void objectRegionGrowing(pcl::PointCloud<PointNormal>::Ptr cloud, std::vector<PlaneWithObjInd> &objects, int max_object_size);
    pcl::PointCloud<PointNormal>::Ptr fromObjectVecToObjectCloud(const std::vector<PlaneWithObjInd> objects, pcl::PointCloud<PointNormal>::Ptr cloud);
    void saveObjectsWithPlanes(std::string path, const std::vector<PlaneWithObjInd> objects, pcl::PointCloud<PointNormal>::Ptr cloud);
    void mergeObjects(std::vector<PlaneWithObjInd>& objects);
    std::vector<double> filterBasedOnColor(std::vector<PlaneWithObjInd>& objects, pcl::PointCloud<PointNormal>::Ptr cloud, int _nr_binsh=10);
    std::vector<int> removePlanarObjects (std::vector<PlaneWithObjInd>& objects, pcl::PointCloud<PointNormal>::Ptr cloud, float _plane_dist_thr=0.005);
    int checkPlanarity (PlaneWithObjInd& objects, pcl::PointCloud<PointNormal>::Ptr cloud, float _plane_dist_thr);
    void filterPlanarAndColor(std::vector<PlaneWithObjInd>& objects, pcl::PointCloud<PointNormal>::Ptr cloud, std::string path, float _plane_dist_thr=0.005, int _nr_bins=10 );
    double checkColorSimilarity(PlaneWithObjInd& object, pcl::PointCloud<PointNormal>::Ptr cloud, int _nr_bins=10) ;
    void cleanResult(std::vector<DetectedObject> &detected_objects);
};

#endif // CHANGE_DETECTION_H
