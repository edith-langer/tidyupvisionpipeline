#include "object_visualization.h"

ObjectVisualization::ObjectVisualization(pcl::PointCloud<PointNormal>::Ptr ref_cloud, pcl::PointCloud<PointNormal>::Ptr curr_cloud,
                                         std::vector<DetectedObject>  disappeared_objects,  std::vector<DetectedObject> novel_objects,
                                          std::vector<DetectedObject> displaced_ref_objects,  std::vector<DetectedObject> displaced_curr_objects)
{
    assert(displaced_curr_objects.size() == displaced_ref_objects.size());

    this->ref_cloud = ref_cloud;
    this->disappeared_objects = disappeared_objects;
    this->curr_cloud = curr_cloud;
    this->novel_objects = novel_objects;
    this->displaced_ref_objects = displaced_ref_objects;
    this->displaced_curr_objects = displaced_curr_objects;
}

void ObjectVisualization::visualize() {
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
                new pcl::visualization::PCLVisualizer("3D Viewer"));

    pcl::PointCloud<PointRGBA>::Ptr disappeared_objects_cloud(new pcl::PointCloud<PointRGBA>);
    pcl::PointCloud<PointRGBA>::Ptr novel_objects_cloud(new pcl::PointCloud<PointRGBA>);
    pcl::PointCloud<PointRGBA>::Ptr displaced_ref_objects_cloud(new pcl::PointCloud<PointRGBA>);
    pcl::PointCloud<PointRGBA>::Ptr displaced_curr_objects_cloud(new pcl::PointCloud<PointRGBA>);

    //prepare clouds to highlight detected objects
    //Removed objects : RED
    for (DetectedObject o : disappeared_objects) {
        pcl::PointCloud<PointRGBA>::Ptr o_cloud(new pcl::PointCloud<PointRGBA>);
        pcl::copyPointCloud(*o.object_cloud_, *o_cloud);
        for (size_t i = 0; i < o_cloud->size(); i++) {
            o_cloud->points[i].r = 255;
            o_cloud->points[i].g = 0;
            o_cloud->points[i].b = 0;
            o_cloud->points[i].a = 128;
        }
        *disappeared_objects_cloud += *o_cloud;
    }

    //New objects: GREEN
    for (DetectedObject o : novel_objects) {
        pcl::PointCloud<PointRGBA>::Ptr o_cloud(new pcl::PointCloud<PointRGBA>);
        pcl::copyPointCloud(*o.object_cloud_, *o_cloud);
        for (size_t i = 0; i < o_cloud->size(); i++) {
            o_cloud->points[i].r = 0;
            o_cloud->points[i].g = 255;
            o_cloud->points[i].b = 0;
            o_cloud->points[i].a = 128;
        }
        *novel_objects_cloud += *o_cloud;
    }

    //we need different color values for the displaced objects
    //Displaced objects: different colors (r and g random and b high number)
    for (size_t o = 0; o < displaced_ref_objects.size(); o++) {
        const DetectedObject &ref_object = displaced_ref_objects[o];
        const DetectedObject &curr_object = displaced_curr_objects[o];
        pcl::PointCloud<PointRGBA>::Ptr ref_objects_cloud(new pcl::PointCloud<PointRGBA>);
        pcl::PointCloud<PointRGBA>::Ptr curr_objects_cloud(new pcl::PointCloud<PointRGBA>);
        pcl::copyPointCloud(*ref_object.object_cloud_, *ref_objects_cloud);
        pcl::copyPointCloud(*curr_object.object_cloud_, *curr_objects_cloud);
        uint r = rand() % 200;
        uint g = rand() % 200;
        uint range = 256 - 100;
        uint b = rand() % range + 100; //between 100 and 255
        for (size_t i = 0; i < ref_objects_cloud->size(); i++) {
            ref_objects_cloud->points[i].r = r;
            ref_objects_cloud->points[i].g = g;
            ref_objects_cloud->points[i].b = b;
            ref_objects_cloud->points[i].a = 100;
        }
        for (size_t i = 0; i < curr_objects_cloud->size(); i++) {
            curr_objects_cloud->points[i].r = r;
            curr_objects_cloud->points[i].g = g;
            curr_objects_cloud->points[i].b = b;
            curr_objects_cloud->points[i].a = 100;
        }
        *displaced_ref_objects_cloud += *ref_objects_cloud;
        *displaced_curr_objects_cloud += *curr_objects_cloud;
    }



    //prepare clouds to highlight detected objects
    int v1, v2;
    //                     xmin ymin xmax ymax
    viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
    viewer->createViewPort(0.51, 0.0, 1.0, 1.0, v2);

    viewer->addPointCloud<PointNormal>(ref_cloud, "reference scene", v1);
    pcl::visualization::PointCloudColorHandlerRGBAField<PointRGBA> rgba(disappeared_objects_cloud);
    viewer->addPointCloud<PointRGBA>(disappeared_objects_cloud, rgba, "disappeared objects", v1);
    rgba.setInputCloud(displaced_ref_objects_cloud);
    viewer->addPointCloud<PointRGBA>(displaced_ref_objects_cloud, rgba, "displaced objects in reference scene", v1);
    viewer->addPointCloud<PointNormal>(curr_cloud, "current scene", v2);
    rgba.setInputCloud(novel_objects_cloud);
    viewer->addPointCloud<PointRGBA>(novel_objects_cloud, rgba, "novel objects", v2);
    rgba.setInputCloud(displaced_curr_objects_cloud);
    viewer->addPointCloud<PointRGBA>(displaced_curr_objects_cloud, rgba, "displaced objects in current scene", v2);

    //viewer->resetCameraViewpoint("reference scene");
    //viewer->resetCameraViewpoint("current scene");
    viewer->resetCamera();

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }


}
