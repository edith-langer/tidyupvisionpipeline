#ifndef DETECTED_OBJECT_H
#define DETECTED_OBJECT_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>

#include "plane_object_extraction.h"

typedef pcl::PointXYZRGBNormal PointNormal;

const float max_dist_for_being_static = 0.1; //how much can the object be displaced to still count as static

enum ObjectState {NEW, REMOVED, DISPLACED, STATIC, UNKNOWN};

struct Match {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix<float,4,4,Eigen::DontAlign> transform;
    int model_id;
    int object_id;
    float confidence;

    Match() {model_id = -1; object_id = -1, confidence = -1; transform = Eigen::Matrix4f::Identity();}
    Match(int m_id, int o_id, float conf, Eigen::Matrix4f t) {model_id = m_id; object_id = o_id; confidence = conf; transform = Eigen::Matrix4f(t);}
};

struct DetectedObject {


public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    DetectedObject(){}

    DetectedObject(pcl::PointCloud<PointNormal>::Ptr object_cloud, pcl::PointCloud<PointNormal>::Ptr plane_cloud,
                   PlaneStruct supp_plane, ObjectState object_state = UNKNOWN, std::string object_folder_path = "") : unique_id_(++s_id), object_cloud_(object_cloud),
                    plane_cloud_(plane_cloud), supp_plane_(supp_plane), state_(object_state), object_folder_path_(object_folder_path) {}

    Match match_; //this is only relevant for displaced objects
    pcl::PointCloud<PointNormal>::Ptr object_cloud_;
    pcl::PointCloud<PointNormal>::Ptr plane_cloud_;
    PlaneStruct supp_plane_;
    std::string object_folder_path_;
    ObjectState state_;

    int getID() const {return unique_id_;}
    
protected:
    static int s_id;

private:
    int unique_id_;
};

#endif // DETECTED_OBJECT_H
