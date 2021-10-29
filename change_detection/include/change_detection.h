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
#include <PPFRecognizerParameter.h>

#include "scene_differencing_points.h"
#include "local_object_verification.h"
#include "object_visualization.h"
#include "region_growing.h"
#include "color_histogram.h"
#include "detected_object.h"
#include "object_matching.h"
#include "plane_object_extraction.h"
#include "color_histogram.h"

#include "settings.h"
#include "mathhelpers.h"

typedef pcl::PointXYZRGBNormal PointNormal;
typedef pcl::PointXYZRGBL PointLabel;


class ChangeDetection
{
public:
    ChangeDetection(std::string ppf_config_path) : ppf_config_path_(ppf_config_path) {
        curr_checked_plane_point_cloud_.reset(new pcl::PointCloud<PointNormal>);
        ref_checked_plane_point_cloud_.reset(new pcl::PointCloud<PointNormal>);
    }

    void init(pcl::PointCloud<PointNormal>::Ptr ref_cloud, pcl::PointCloud<PointNormal>::Ptr curr_cloud,
              const Eigen::Vector4f &ref_plane_coeffs, const Eigen::Vector4f &curr_plane_coeffs,
              pcl::PointCloud<pcl::PointXYZ>::Ptr ref_convex_hull_pts, pcl::PointCloud<pcl::PointXYZ>::Ptr curr_convex_hull_pts,
              std::string ppf_model_path, std::string output_path, std::string merge_object_parts_folder, bool perform_LV_matching = true);

    void setOutputPath (std::string output_path) {
        output_path_ = output_path;
    }

    void setPPFModelPath (std::string model_path) {
        ppf_model_path_ = model_path;
    }

    void setRefCloud(pcl::PointCloud<PointNormal>::Ptr cloud) {
        this->ref_cloud_ = cloud;
    }

    void setCurrCloud(pcl::PointCloud<PointNormal>::Ptr cloud) {
        this->curr_cloud_ = cloud;
    }

    void compute(std::vector<DetectedObject> &ref_result, std::vector<DetectedObject> &curr_result);
    std::vector<PlaneWithObjInd> getObjectsFromPlane(pcl::PointCloud<PointNormal>::Ptr input_cloud, Eigen::Vector4f plane_coeffs,
                                                     pcl::PointCloud<pcl::PointXYZ>::Ptr convex_hull_pts,
                                                     pcl::PointCloud<PointNormal>::Ptr prev_checked_plane_cloud, std::string res_path);
    void objectRegionGrowing(pcl::PointCloud<PointNormal>::Ptr cloud, std::vector<PlaneWithObjInd> &objects, int max_object_size=std::numeric_limits<int>::max()); // std::numeric_limits<int>::max());
    void mergeObjects(std::vector<PlaneWithObjInd>& objects);
    pcl::PointCloud<PointNormal>::Ptr fromObjectVecToObjectCloud(const std::vector<PlaneWithObjInd> objects, pcl::PointCloud<PointNormal>::Ptr cloud, bool keepOrganized=true);
    pcl::PointCloud<PointNormal>::Ptr fromDetObjectVecToCloud(const std::vector<DetectedObject> object_vec, bool withStaticObjects=true);
    void upsampleObjectsAndPlanes(pcl::PointCloud<PointNormal>::Ptr orig_cloud, pcl::PointCloud<PointNormal>::Ptr ds_cloud,
                                  std::vector<PlaneWithObjInd> &objects, double leaf_size, std::string res_path);
    void upsampleObjectsAndPlanes(pcl::PointCloud<PointNormal>::Ptr orig_cloud, std::vector<DetectedObject> &objects, double leaf_size, std::string res_path);
    std::tuple<pcl::PointCloud<PointNormal>::Ptr, std::vector<int> > upsampleObjects(pcl::octree::OctreePointCloudSearch<PointNormal>::Ptr octree, pcl::PointCloud<PointNormal>::ConstPtr orig_input_cloud,
                                                                                     pcl::PointCloud<PointNormal>::ConstPtr objects_ds_cloud, std::string output_path, int counter);

    DetectedObject fromPlaneIndObjToDetectedObject (pcl::PointCloud<PointNormal>::Ptr curr_cloud, PlaneWithObjInd obj);
    void performLV(std::vector<DetectedObject> &ref_objects, std::vector<DetectedObject> &curr_objects);


    static void mergeObjectParts(std::vector<DetectedObject> &detected_objects, std::string merge_object_parts_folder);
    static void filterSmallVolumes(std::vector<DetectedObject> &objects, double volume_thr, int min_obj_size=0);
    static void filterUnwantedObjects(std::vector<DetectedObject> &objects, double volume_thr=0, int min_obj_size=0, int max_obj_size=std::numeric_limits<int>::max(),
                                      double plane_dist_thr =0.01, double plane_thr =0.9, std::string save_path="");
    static void refineNormals(pcl::PointCloud<PointNormal>::Ptr object_cloud);

private:
    //std::string object_store_path_; //the model objects and their ppf model get stored here --> for now we store everything in output path
    std::string ppf_config_path_; //the config file that stores the parrameters for PPF
    std::string output_path_; //all debugging things will get stored there (+model objects and their ppf model)
    std::string ppf_model_path_;
    std::string merge_object_parts_path_;

    std::vector<PlaneWithObjInd> potential_objects_;
    pcl::PointCloud<PointNormal>::Ptr ref_cloud_;
    pcl::PointCloud<PointNormal>::Ptr curr_cloud_;
    Eigen::Vector4f ref_plane_coeffs_;
    Eigen::Vector4f curr_plane_coeffs_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr curr_convex_hull_pts_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr ref_convex_hull_pts_;

    bool do_LV_before_matching_;

    pcl::PointCloud<PointNormal>::Ptr curr_checked_plane_point_cloud_;
    pcl::PointCloud<PointNormal>::Ptr ref_checked_plane_point_cloud_;


    void saveObjectsWithPlanes(std::string path, const std::vector<PlaneWithObjInd> objects, pcl::PointCloud<PointNormal>::Ptr cloud);
    void saveObjectsWithPlanes(std::string path, const std::vector<DetectedObject> objects);
    std::vector<double> filterBasedOnColor(std::vector<PlaneWithObjInd>& objects, pcl::PointCloud<PointNormal>::Ptr cloud, int _nr_binsh=10);
    std::vector<int> removePlanarObjects (std::vector<PlaneWithObjInd>& objects, pcl::PointCloud<PointNormal>::Ptr cloud, float _plane_dist_thr=0.01);
    int checkPlanarity (PlaneWithObjInd& objects, pcl::PointCloud<PointNormal>::Ptr cloud, float _plane_dist_thr);
    void filterPlanarAndColor(std::vector<PlaneWithObjInd>& objects, pcl::PointCloud<PointNormal>::Ptr cloud, std::string path, float _plane_dist_thr=0.01, int _nr_bins=10 );
    double checkColorSimilarityHistogram(PlaneWithObjInd& object, pcl::PointCloud<PointNormal>::Ptr cloud, std::string path="", int _nr_bins=10) ;
    void cleanResult(std::vector<DetectedObject> &detected_objects);
    void matchAndRemoveObjects (pcl::PointCloud<PointNormal>::Ptr remaining_scene_points, pcl::PointCloud<PointNormal>::Ptr full_object_cloud, std::vector<PlaneWithObjInd> &extracted_objects);
    int checkVerticalPlanarity(PlaneWithObjInd& object, pcl::PointCloud<PointNormal>::Ptr cloud, float _plane_dist_thr);
    void filterVerticalPlanes(std::vector<PlaneWithObjInd>& objects, pcl::PointCloud<PointNormal>::Ptr cloud, std::string path, float _plane_dist_thr=0.005);

};

#endif // CHANGE_DETECTION_H
