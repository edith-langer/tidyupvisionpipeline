#ifndef LOCAL_OBJECT_VERIFICATION_H
#define LOCAL_OBJECT_VERIFICATION_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>

//#include <pcl/registration/gicp.h>
#include <pcl/registration/warp_point_rigid_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_lm.h>

#include <PPFRecognizerParameter.h>

#include "plane_object_extraction.h"
#include "warp_point_rigid_4d.h"
#include "scene_differencing_points.h"
#include "color_histogram.h"
#include "object_matching.h"

typedef pcl::PointXYZRGBNormal PointNormal;

struct LocalObjectVerificationParams {
    float diff_dist =0.02;
    float add_crop_static = 0.2;
    float icp_max_corr_dist = 0.15;
    float icp_max_corr_dist_plane = 0.05;
    int icp_max_iter = 500;
    float icp_ransac_thr = 0.009;
    float color_weight = 0.25;
    float overlap_weight = 0.75;
    float min_score_thr = 0.7;
};

class LocalObjectVerification
{
public:
    LocalObjectVerification(pcl::PointCloud<PointNormal>::Ptr ref_object, //pcl::PointCloud<PointNormal>::Ptr ref_plane,
                            pcl::PointCloud<PointNormal>::Ptr curr_object, //pcl::PointCloud<PointNormal>::Ptr curr_plane,
                            LocalObjectVerificationParams params);
    std::tuple<std::vector<int>, std::vector<int> > computeLV();
    void setDebugOutputPath (std::string debug_output_path);

    void setRefCloud(pcl::PointCloud<PointNormal>::Ptr cloud) {
        this->ref_object_ = cloud;
    }

    void setCurrCloud(pcl::PointCloud<PointNormal>::Ptr cloud) {
        this->curr_object_ = cloud;
    }

private:
    pcl::PointCloud<PointNormal>::Ptr ref_object_;
    pcl::PointCloud<PointNormal>::Ptr curr_object_;
    pcl::PointCloud<PointNormal>::Ptr ref_plane_;
    pcl::PointCloud<PointNormal>::Ptr curr_plane_;

    LocalObjectVerificationParams params_;

    std::string debug_output_path;
};

#endif // LOCAL_OBJECT_VERIFICATION_H
