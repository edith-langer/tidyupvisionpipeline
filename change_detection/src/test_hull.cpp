#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <chrono>

#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/crop_hull.h>

typedef pcl::PointXYZRGBNormal PointNormal;
typedef pcl::PointXYZRGB PointRGB;
typedef pcl::PointXYZRGBL PointLabel;

using namespace  std;

template <typename T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i != v.size() - 1)
            os << ", ";
    }
    os << "]\n";
    return os;
}

/// To be able to use isPointIn2DPolygon() the point has to be projected onto the plane. Also for using cropHull
/// Code to do so found here https://github.com/PointCloudLibrary/pcl/blob/master/apps/src/organized_segmentation_demo.cpp
/// see also https://stackoverflow.com/questions/9605556/how-to-project-a-point-onto-a-plane-in-3d
//double ptp_dist = std::abs(model.values[0] * pt.x + model.values[1] * pt.y +
//                            model.values[2] * pt.z + model.values[3]);
// if (ptp_dist >= 0.1)
//   return false;

// // project the point onto the plane
// Eigen::Vector3f mc(model.values[0], model.values[1], model.values[2]);
// Eigen::Vector3f pt_vec;
// pt_vec[0] = pt.x;
// pt_vec[1] = pt.y;
// pt_vec[2] = pt.z;
// Eigen::Vector3f projected(pt_vec - mc * float(ptp_dist));
// PointT projected_pt;
// projected_pt.x = projected[0];
// projected_pt.y = projected[1];
// projected_pt.z = projected[2];

int main(int argc, char* argv[])
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr hullCloud(new pcl::PointCloud<pcl::PointXYZ>());
    std::vector<pcl::Vertices> vertices;
    pcl::Vertices vt;

    // inside point
    inputCloud->push_back(pcl::PointXYZ(3,3,0));
    // hull
    hullCloud->push_back(pcl::PointXYZ(0,0,0));
    hullCloud->push_back(pcl::PointXYZ(10,0,0));
    hullCloud->push_back(pcl::PointXYZ(10,5,0));
    hullCloud->push_back(pcl::PointXYZ(0,5,0));
    hullCloud->push_back(pcl::PointXYZ(0,0,0));
    // outside point
    inputCloud->push_back(pcl::PointXYZ(-3,-3,0));

    pcl::io::savePCDFile("/home/edith/Desktop/input.pcd", *inputCloud);
    vt.vertices.push_back(1);
    vt.vertices.push_back(2);
    vt.vertices.push_back(3);
    vt.vertices.push_back(4);
    vt.vertices.push_back(5);
    vertices.push_back(vt);

    pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::CropHull<pcl::PointXYZ> cropHull;
    cropHull.setHullIndices(vertices);
    cropHull.setHullCloud(hullCloud);
    cropHull.setInputCloud(inputCloud);
    cropHull.setDim(2);
    cropHull.setCropOutside(true);

    std::vector<int> indices;
    cropHull.filter(indices);
    cropHull.filter(*outputCloud);

    std::cout << indices;


    pcl::io::savePCDFile("/home/edith/Desktop/filtered.pcd", *outputCloud);
}


