#ifndef DETECTED_OBJECT_H
#define DETECTED_OBJECT_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>

#include "plane_object_extraction.h"

typedef pcl::PointXYZRGBNormal PointNormal;

struct DetectedObject {
protected:
    static int s_id;

public:
    DetectedObject(pcl::PointCloud<PointNormal>::Ptr object_cloud, pcl::PointCloud<PointNormal>::Ptr plane_cloud,
                   PlaneStruct supp_plane) : unique_id_(++s_id), object_cloud_(object_cloud),
                    plane_cloud_(plane_cloud), supp_plane_(supp_plane), object_folder_path_("") {}

    pcl::PointCloud<PointNormal>::Ptr object_cloud_;
    pcl::PointCloud<PointNormal>::Ptr plane_cloud_;
    PlaneStruct supp_plane_;
    std::string object_folder_path_;

    int getID() {return unique_id_;}

private:
    int unique_id_;
};

#endif // DETECTED_OBJECT_H
