#ifndef OBJECT_VISUALIZATION_H
#define OBJECT_VISUALIZATION_H

#include <thread>
#include <chrono>
#include <stdlib.h>

#include <pcl/visualization/pcl_visualizer.h>

#include "detected_object.h"

typedef pcl::PointXYZRGBNormal PointNormal;
typedef pcl::PointXYZRGBA PointRGBA;

class ObjectVisualization
{
public:
    ObjectVisualization(pcl::PointCloud<PointNormal>::Ptr ref_cloud, pcl::PointCloud<PointNormal>::Ptr curr_cloud,
                        std::vector<DetectedObject> disappeared_objects, std::vector<DetectedObject> novel_objects,
                        std::vector<DetectedObject> displaced_ref_objects, std::vector<DetectedObject> displaced_curr_objects);

    void visualize();
private:
    pcl::PointCloud<PointNormal>::Ptr ref_cloud;
    pcl::PointCloud<PointNormal>::Ptr curr_cloud;
    std::vector<DetectedObject> disappeared_objects;
    std::vector<DetectedObject> novel_objects;
    std::vector<DetectedObject> displaced_ref_objects;
    std::vector<DetectedObject> displaced_curr_objects;
};

#endif // OBJECT_VISUALIZATION_H
