#ifndef OBJECT_VISUALIZATION_H
#define OBJECT_VISUALIZATION_H

#include <thread>
#include <chrono>

#include <pcl/visualization/pcl_visualizer.h>


typedef pcl::PointXYZRGBNormal PointType;

class ObjectVisualization
{
public:
    ObjectVisualization(pcl::PointCloud<PointType>::Ptr ref_cloud, pcl::PointCloud<PointType>::Ptr disappeared_objects,
                         pcl::PointCloud<PointType>::Ptr curr_cloud, pcl::PointCloud<PointType>::Ptr novel_objects);

    void visualize();
private:
    pcl::PointCloud<PointType>::Ptr ref_cloud;
    pcl::PointCloud<PointType>::Ptr curr_cloud;
    pcl::PointCloud<PointType>::Ptr disappeared_objects;
    pcl::PointCloud<PointType>::Ptr novel_objects;
};

#endif // OBJECT_VISUALIZATION_H
