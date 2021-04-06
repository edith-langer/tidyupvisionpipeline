#ifndef LOCAL_OBJECT_VERIFICATION_H
#define LOCAL_OBJECT_VERIFICATION_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>

//#include <pcl/registration/gicp.h>
#include <pcl/registration/warp_point_rigid_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_lm.h>

#include "plane_object_extraction.h"
#include "warp_point_rigid_4d.h"
#include "scene_differencing_points.h"
#include "color_histogram.h"

typedef pcl::PointXYZRGBNormal PointType;

struct LocalObjectVerificationParams {
    float diff_dist =0.02;
    float add_crop_static = 0.2;
    float icp_max_corr_dist = 0.15;
    float icp_max_corr_dist_plane = 0.05;
    int icp_max_iter = 500;
    float icp_ransac_thr = 0.009;
    float color_weight = 0.25;
    float overlap_weight = 0.75;
    float min_score_thr = 0.6;
};

class LocalObjectVerification
{
public:
    LocalObjectVerification(std::vector<PlaneWithObjInd> potential_objects, pcl::PointCloud<PointType>::Ptr ref_cloud,
                            pcl::PointCloud<PointType>::Ptr curr_cloud, LocalObjectVerificationParams params);
    std::vector<PlaneWithObjInd> verify_changed_objects();
    void setDebugOutputPath (std::string debug_output_path);

    void setRefCloud(pcl::PointCloud<PointType>::Ptr cloud) {
        this->ref_cloud = cloud;
    }

    void setCurrCloud(pcl::PointCloud<PointType>::Ptr cloud) {
        this->curr_cloud = cloud;
    }

    void setPotentialObjects(std::vector<PlaneWithObjInd> potential_objects) {
        this->potential_objects = potential_objects;
    }


private:
    std::vector<PlaneWithObjInd> potential_objects;
    pcl::PointCloud<PointType>::Ptr ref_cloud;
    pcl::PointCloud<PointType>::Ptr curr_cloud;

    LocalObjectVerificationParams params_;

    std::string debug_output_path;
};

#endif // LOCAL_OBJECT_VERIFICATION_H
