#ifndef EXTRACTOBJECTSUSINGSEMANTICS_H
#define EXTRACTOBJECTSUSINGSEMANTICS_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>

#include "plane_object_extraction.h"

typedef pcl::PointXYZRGB PointRGB;

class ExtractObjectsFromPlane
{
public:
    ExtractObjectsFromPlane(pcl::PointCloud<PointNormal>::Ptr cloud_, Eigen::Vector4f main_plane_coeffs,
                                 std::string output_path="");

    void setDebugOutputPath (std::string debug_output_path_);


    void setCloud(pcl::PointCloud<PointRGB>::Ptr cloud) {
        this->cloud_ = cloud;
    }
    pcl::PointCloud<PointRGB>::Ptr getCloud() {
        return this->cloud_;
    }

    std::vector<PlaneWithObjInd> computeObjectsOnPlanes(); //this method returns potential object (points) that correspond to new or disappeard objects supported ba a plane or semantically segmented
    std::vector<pcl::PointIndices> cleanResult(pcl::PointCloud<PointRGB>::Ptr cloud_, float cluster_thr);

private:
    pcl::PointCloud<PointRGB>::Ptr cloud_;
    std::string debug_output_path_;
    Eigen::Vector4f main_plane_coeffs_;

};

#endif // EXTRACTOBJECTSUSINGSEMANTICS_H
