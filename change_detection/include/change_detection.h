#ifndef CHANGE_DETECTION_H
#define CHANGE_DETECTION_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>

#include <pcl/registration/warp_point_rigid_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_lm.h>

#include "plane_object_extraction.h"
#include "warp_point_rigid_4d.h"
#include "scene_differencing_points.h"
#include "color_histogram.h"

typedef pcl::PointXYZRGBNormal PointNormal;
typedef pcl::PointXYZRGBL PointLabel;

struct ChangeDetectionResult {
    std::vector<std::vector<int> > new_objects;
    std::vector<std::vector<int> > removed_objects;
    std::vector<std::vector<int> > static_objects;
    std::vector<std::vector<int> > displaced_objects;
};


class ChangeDetection
{
public:
    ChangeDetection(pcl::PointCloud<PointNormal>::Ptr ref_cloud, pcl::PointCloud<PointNormal>::Ptr curr_cloud,
                    Eigen::Vector4f ref_plane_coeffs, Eigen::Vector4f curr_plane_coeffs,
                    std::vector<pcl::PointXYZ> ref_convex_hull_pts, std::vector<pcl::PointXYZ> curr_convex_hull_pts,
                    std::string output_path = "") :
        ref_cloud_(ref_cloud), curr_cloud_(curr_cloud), ref_plane_coeffs_(ref_plane_coeffs),
        curr_plane_coeffs_(curr_plane_coeffs),  curr_convex_hull_pts_(curr_convex_hull_pts),
        ref_convex_hull_pts_(ref_convex_hull_pts),output_path_(output_path) {}


    void setOutputPath (std::string output_path) {
        output_path_ = output_path;
    }

    void setRefCloud(pcl::PointCloud<PointNormal>::Ptr cloud) {
        this->ref_cloud_ = cloud;
    }

    void setCurrCloud(pcl::PointCloud<PointNormal>::Ptr cloud) {
        this->curr_cloud_ = cloud;
    }

    void compute(ChangeDetectionResult& ref_result, ChangeDetectionResult& curr_result);

private:
    std::string object_store_path_; //the model objects and their ppf model get stored here

    std::vector<PlaneWithObjInd> potential_objects_;
    pcl::PointCloud<PointNormal>::Ptr ref_cloud_;
    pcl::PointCloud<PointNormal>::Ptr curr_cloud_;

    Eigen::Vector4f ref_plane_coeffs_;
    Eigen::Vector4f curr_plane_coeffs_;
    std::vector<pcl::PointXYZ> curr_convex_hull_pts_;
    std::vector<pcl::PointXYZ> ref_convex_hull_pts_;

    std::string output_path_;

    bool do_LV_before_matching = false;


    void refineNormals(pcl::PointCloud<PointNormal>::Ptr object_cloud);
    std::vector<pcl::PointIndices> cleanResult(pcl::PointCloud<PointNormal>::Ptr cloud, float cluster_thr);
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
};

#endif // CHANGE_DETECTION_H
