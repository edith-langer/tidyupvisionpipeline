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

#include "settings.h"

typedef pcl::PointXYZRGBNormal PointNormal;


struct LocalObjectVerificationParams {
    float diff_dist =0.015;
    float add_crop_static = 0.2;
    float icp_max_corr_dist = 0.15;
    float icp_max_corr_dist_plane = 0.05;
    int icp_max_iter = 500;
    float icp_ransac_thr = 0.009;
    float color_weight = 0.25;
    float overlap_weight = 0.75;
    float min_score_thr = 0.7;
};

struct LVResult {
    std::vector<int> obj_non_matching_pts;
    std::vector<int> model_non_matching_pts;
    Eigen::Matrix<float,4,4,Eigen::DontAlign> transform_obj_to_model;
    FitnessScoreStruct fitness_score;
};

class LocalObjectVerification
{
public:
    LocalObjectVerification(pcl::PointCloud<PointNormal>::ConstPtr ref_object, //pcl::PointCloud<PointNormal>::Ptr ref_plane,
                            pcl::PointCloud<PointNormal>::ConstPtr curr_object, //pcl::PointCloud<PointNormal>::Ptr curr_plane,
                            LocalObjectVerificationParams params) : ref_object_(ref_object), curr_object_(curr_object), params_(params) {
        this->debug_output_path="";
    }
    LVResult computeLV();
    void setDebugOutputPath (std::string debug_output_path);

    void setRefCloud(pcl::PointCloud<PointNormal>::Ptr cloud) {
        this->ref_object_ = cloud;
    }

    void setCurrCloud(pcl::PointCloud<PointNormal>::Ptr cloud) {
        this->curr_object_ = cloud;
    }

private:
    pcl::PointCloud<PointNormal>::ConstPtr ref_object_;
    pcl::PointCloud<PointNormal>::ConstPtr curr_object_;
    pcl::PointCloud<PointNormal>::Ptr ref_plane_;
    pcl::PointCloud<PointNormal>::Ptr curr_plane_;

    LocalObjectVerificationParams params_;

    std::string debug_output_path;
};

#endif // LOCAL_OBJECT_VERIFICATION_H
