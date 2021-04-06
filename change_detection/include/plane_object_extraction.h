#ifndef PLANE_OBJECT_EXTRACTION_H
#define PLANE_OBJECT_EXTRACTION_H

#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <limits>
#include <algorithm>

#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/search/impl/search.hpp>

typedef pcl::PointXYZRGBNormal PointNormal;
typedef pcl::PointXYZRGB PointRGB;

struct PlaneStruct {
    pcl::PointIndices::Ptr plane_ind;
    pcl::ModelCoefficients::Ptr coeffs;
    float avg_z;

    PlaneStruct(){}
    PlaneStruct(pcl::PointIndices::Ptr ind, pcl::ModelCoefficients::Ptr c, float z) : plane_ind(ind), coeffs(c), avg_z(z) {}

    bool operator < (const PlaneStruct& plane) const
    {
        //std::cout<<avg_z << " " << plane.avg_z << std::endl;
        //return (-coeffs->values[3]/coeffs->values[2] < -plane.coeffs->values[3]/plane.coeffs->values[2]);
        return avg_z < plane.avg_z;
    }

    PlaneStruct& operator=(const PlaneStruct rhs) {
        plane_ind = boost::make_shared<pcl::PointIndices>(*(rhs.plane_ind));
        avg_z =  rhs.avg_z;
        coeffs = boost::make_shared<pcl::ModelCoefficients>(*(rhs.coeffs));
        return *this;
    }
};


struct PlaneWithObjInd {
    PlaneStruct plane;
    std::vector<int> obj_indices;
};

class ExtractObjectsFromPlanes {
public:
    ExtractObjectsFromPlanes(pcl::PointCloud<PointNormal>::Ptr cloud, Eigen::Vector4f main_plane_coeffs, std::vector<pcl::PointXYZ> convex_hull_pts, std::string output_path="");
    ~ExtractObjectsFromPlanes();

    void setResultPath(std::string path);

    PlaneWithObjInd computeObjectsOnPlanes(); //this method returns potential object (points) that correspond to new or disappeard objects supported by a plane


private:
    pcl::PointCloud<PointNormal>::Ptr orig_cloud_;
    Eigen::Vector4f main_plane_coeffs_;
    std::vector<pcl::PointXYZ> convex_hull_pts_;
    std::string result_path_;
    std::vector<PlaneStruct> hor_planes_;

    PlaneWithObjInd extractObjectInd();
    void filter_flying_objects(pcl::PointCloud<PointNormal>::Ptr orig_cloud_, pcl::PointIndices::Ptr ind, float avg_plane_z);
};

#endif //PLANE_OBJECT_EXTRACTION_H
