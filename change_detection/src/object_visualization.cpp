#include "object_visualization.h"

ObjectVisualization::ObjectVisualization(pcl::PointCloud<PointType>::Ptr ref_cloud, pcl::PointCloud<PointType>::Ptr disappeared_objects,
                                         pcl::PointCloud<PointType>::Ptr curr_cloud, pcl::PointCloud<PointType>::Ptr novel_objects)
{
    assert(ref_cloud->size() == disappeared_objects->size() && curr_cloud->size() == novel_objects->size());

    this->ref_cloud = ref_cloud;
    this->disappeared_objects = disappeared_objects;
    this->curr_cloud = curr_cloud;
    this->novel_objects = novel_objects;
}

void ObjectVisualization::visualize() {
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
                new pcl::visualization::PCLVisualizer("3D Viewer"));

    //prepare clouds to highlight detected objects
    for (size_t i = 0; i < disappeared_objects->size(); i++) {
        if (pcl::isFinite(disappeared_objects->points[i])) {
            ref_cloud->points[i].r = 255;
            ref_cloud->points[i].g = 0;
            ref_cloud->points[i].b = 0;
        }
    }
    for (size_t i = 0; i < novel_objects->size(); i++) {
        if (pcl::isFinite(novel_objects->points[i])) {
            curr_cloud->points[i].r = 0;
            curr_cloud->points[i].g = 0;
            curr_cloud->points[i].b = 255;
        }
    }

    int v1, v2;
    //                     xmin ymin xmax ymax
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->createViewPort(0.51, 0.0, 1.0, 1.0, v2);

    viewer->addPointCloud<PointType>(ref_cloud, "reference scene", v1);
    viewer->addPointCloud<PointType>(curr_cloud, "current scene", v2);

    //viewer->resetCameraViewpoint("reference scene");
    //viewer->resetCameraViewpoint("current scene");
    viewer->resetCamera();

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }


}
