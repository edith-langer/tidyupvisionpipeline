#ifndef MATHHELPERS_H
#define MATHHELPERS_H

#include "pcl/ModelCoefficients.h"
#include "pcl/filters/voxel_grid.h"
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/io/pcd_io.h>

typedef pcl::PointXYZRGBNormal PointNormal;

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


// v4r/common/Downsampler.cpp has a more advanced method if normals are given
inline pcl::PointCloud<PointNormal>::Ptr downsampleCloudVG(pcl::PointCloud<PointNormal>::Ptr input, double leafSize)
{
    std::cout << "PointCloud before filtering has: " << input->points.size () << " data points." << std::endl;

    // Create the filtering object: downsample the dataset using a leaf size
    pcl::VoxelGrid<PointNormal> vg;
    pcl::PointCloud<PointNormal>::Ptr cloud_filtered (new pcl::PointCloud<PointNormal>);
    vg.setInputCloud (input);
    vg.setLeafSize (leafSize, leafSize, leafSize);
    vg.setDownsampleAllData(true);
    vg.filter (*cloud_filtered);
    std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl;

    return cloud_filtered;
}

inline bool isObjectPlanar(pcl::PointCloud<PointNormal>::ConstPtr object, float plane_dist_thr, float plane_acc_thr) {
    pcl::SACSegmentation<PointNormal> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (plane_dist_thr);
    seg.setInputCloud (object);
    seg.segment (*inliers, *coefficients);

    if (inliers->indices.size() == 0)
        return false;
    //I don't know why this is happening, but it is apprently not a correct solution and would include all points as plane inliers
    if (coefficients->values[0] == 0 && coefficients->values[1] == 0 && coefficients->values[2] == 0 && coefficients->values[3] == 0)
        return false;

    if (inliers->indices.size()/object->size() > plane_acc_thr)
        return true;
    return false;
}

inline bool isObjectUnwanted(const pcl::PointCloud<PointNormal>::ConstPtr &object, double volume_thr, int min_obj_size, int max_obj_size,
                             double plane_dist_thr, double plane_thr, std::string save_filename="") {
    //does the object have enough points?
    if (object->size() < min_obj_size) {
        if (save_filename != "")
            pcl::io::savePCDFileBinary(save_filename + "_toosmall.pcd", *object);
        return true;
    }
    //does the object have too many points?
    if (object->size() > max_obj_size) {
        if (save_filename != "")
            pcl::io::savePCDFileBinary(save_filename + "_toobig.pcd", *object);
        return true;
    }
    //is the object planar?
    if (isObjectPlanar(object, plane_dist_thr, plane_thr)) {
        if (save_filename != "")
            pcl::io::savePCDFileBinary(save_filename + "_toosplanar.pcd", *object);
        return true;
    }

    //does the object has a certain volume?
    pcl::PointCloud<PointNormal>::Ptr cloud_hull (new pcl::PointCloud<PointNormal>);
    pcl::ConvexHull<PointNormal> chull;
    chull.setInputCloud (object);
    chull.setDimension(3);
    chull.setComputeAreaVolume(true);
    chull.reconstruct (*cloud_hull);
    double volume = chull.getTotalVolume();
    if (volume < volume_thr) {
        if (save_filename != "")
            pcl::io::savePCDFileBinary(save_filename + "_toolittlevolume.pcd", *object);
        return true;
    }

    return false;
}

#endif // MATHHELPERS_H
