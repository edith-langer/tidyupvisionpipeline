#ifndef DETECTED_OBJECT_H
#define DETECTED_OBJECT_H

#include <unordered_set>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <pcl/filters/filter.h>

#include "mathhelpers.h"

typedef pcl::PointXYZRGBNormal PointNormal;


const float ds_voxel_size = 0.005;

enum ObjectState {NEW, REMOVED, DISPLACED, STATIC, UNKNOWN};

struct FitnessScoreStruct {
    FitnessScoreStruct() { object_conf = 0.0f; model_conf = 0.0f;}
    FitnessScoreStruct(float o_c, float m_c, std::vector<int> o_pts, std::vector<int> m_pts ) :
        object_conf{o_c}, model_conf{m_c}, object_overlapping_pts{o_pts}, model_overlapping_pts{m_pts} {}
    float object_conf;
    float model_conf;
    std::vector<int> object_overlapping_pts;
    std::vector<int> model_overlapping_pts;
};

struct Match {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix<float,4,4,Eigen::DontAlign> transform;
    int model_id;
    int object_id;
    FitnessScoreStruct fitness_score;

    Match() {model_id = -1; object_id = -1, transform = Eigen::Matrix4f::Identity();}
    Match(int m_id, int o_id, Eigen::Matrix4f t, FitnessScoreStruct f=FitnessScoreStruct()) {
        model_id = m_id; object_id = o_id; transform = Eigen::Matrix4f(t); fitness_score = f;
    }
};

struct DetectedObject {


public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DetectedObject(){}

    //Constructor when we know the ID (e.g. extracted from a DB)
    DetectedObject(int id, pcl::PointCloud<PointNormal>::Ptr object_cloud, pcl::PointCloud<PointNormal>::Ptr plane_cloud,
                   ObjectState object_state = UNKNOWN, std::string object_folder_path = "") : unique_id_(id), object_cloud_(object_cloud),
        plane_cloud_(plane_cloud), state_(object_state), object_folder_path_(object_folder_path) {
        removeNANsAndDownsample(object_cloud);

    }

    DetectedObject(pcl::PointCloud<PointNormal>::Ptr object_cloud, pcl::PointCloud<PointNormal>::Ptr plane_cloud,
                   ObjectState object_state = UNKNOWN, std::string object_folder_path = "") : unique_id_(++s_id), object_cloud_(object_cloud),
        plane_cloud_(plane_cloud), state_(object_state), object_folder_path_(object_folder_path) {

        removeNANsAndDownsample(object_cloud);
    }

    pcl::PointCloud<PointNormal>::Ptr plane_cloud_;
    std::string object_folder_path_;
    ObjectState state_;
    std::unordered_set<int> already_checked_model_ids;
    Match match_; //this is only relevant for displaced objects

    int getID() const {return unique_id_;}
    void setObjectCloud(pcl::PointCloud<PointNormal>::Ptr object_cloud) {
        pcl::PointCloud<PointNormal>::Ptr obj_wo_nans(new pcl::PointCloud<PointNormal>);
        std::vector<int> nan_ind;
        pcl::removeNaNFromPointCloud(*object_cloud, *obj_wo_nans, nan_ind);

        object_cloud_ = obj_wo_nans;
        object_cloud_->is_dense = true;

        object_cloud_ds_.reset(new pcl::PointCloud<PointNormal>);
        object_cloud_ds_ = downsampleCloudVG(object_cloud_, ds_voxel_size);
        object_cloud_ds_->is_dense = true;
    }
    pcl::PointCloud<PointNormal>::ConstPtr getObjectCloud() const {return object_cloud_;}
    pcl::PointCloud<PointNormal>::ConstPtr getObjectCloudDS() const {return object_cloud_ds_;}

    inline void clearClouds() {
        object_cloud_->clear();
        object_cloud_ds_->clear();
    }

    static void setIDCounter(int latest_id) {
        s_id = latest_id;
    }

    static int getIDCounter() {
        return s_id;
    }
    
protected:
    static int s_id;

private:
    int unique_id_;
    pcl::PointCloud<PointNormal>::Ptr object_cloud_;
    pcl::PointCloud<PointNormal>::Ptr object_cloud_ds_;

    void inline removeNANsAndDownsample(pcl::PointCloud<PointNormal>::Ptr object_cloud) {
        pcl::PointCloud<PointNormal>::Ptr obj_wo_nans(new pcl::PointCloud<PointNormal>);
        std::vector<int> nan_ind;
        pcl::removeNaNFromPointCloud(*object_cloud, *obj_wo_nans, nan_ind);

        object_cloud_ = obj_wo_nans;
        object_cloud_->is_dense = true;

        object_cloud_ds_.reset(new pcl::PointCloud<PointNormal>);
        object_cloud_ds_ = downsampleCloudVG(object_cloud_, ds_voxel_size);
        object_cloud_ds_->is_dense = true;
    }
};

#endif // DETECTED_OBJECT_H
