#include "extract_objects_using_semantics.h"

ExtractObjectsFromPlane::ExtractObjectsFromPlane(pcl::PointCloud<PointNormal>::Ptr cloud,
                                                           Eigen::Vector4f main_plane_coeffs, std::string output_path)
{
    cloud_ = cloud;
    main_plane_coeffs_ = main_plane_coeffs;
    debug_output_path_ = output_path;
}


void ExtractObjectsFromPlane::setDebugOutputPath (std::string debug_output_path) {
    this->debug_output_path_ = debug_output_path;
}

std::vector<PlaneWithObjInd> ExtractObjectsFromPlane::computeObjectsOnPlanes() {

    ExtractObjectsFromPlanes eop(cloud_);
    std::string plane_result_path = debug_output_path_ + "/plane_results/";
    boost::filesystem::create_directories(plane_result_path);
    eop.setResultPath(plane_result_path);
    std::cout << "Extract objects supported by a plane " << std::endl;
    std::vector<PlaneWithObjInd> plane_objects = eop.extractObjectInd(main_plane_coeffs_);

    PointNormal nan_point;
    nan_point.x = nan_point.y = nan_point.z = std::numeric_limits<float>::quiet_NaN();
    pcl::PointCloud<PointNormal>::Ptr object_plane_cloud (new pcl::PointCloud<PointNormal>(cloud_->width, cloud_->height, nan_point));
    for (size_t po = 0; po < plane_objects.size(); po++) {
        const PlaneWithObjInd &plane_object = plane_objects[po];
        const std::vector<int> &object_ind = plane_object.obj_indices;
        for (size_t i = 0; i < object_ind.size(); i++) {
            object_plane_cloud->points.at(object_ind[i]) = cloud_->points.at(object_ind[i]);
        }
    }

    cleanResult(object_plane_cloud, 0.02);
    pcl::io::savePCDFileBinary(debug_output_path_ + "/objects_from_plane.pcd", *object_plane_cloud);

    //TODO cluster detected objects in separate objects

    return plane_objects;
}

//removes small clusters from input cloud and returns all valid clusters
std::vector<pcl::PointIndices> ExtractObjectsFromPlane::cleanResult(pcl::PointCloud<PointNormal>::Ptr cloud, float cluster_thr) {
    //TODO check if cloud is empty after removeing nans

    //clean up small things
    pcl::search::KdTree<PointNormal>::Ptr tree (new pcl::search::KdTree<PointNormal>);
    tree->setInputCloud (cloud);

    pcl::EuclideanClusterExtraction<PointNormal> ec;
    std::vector<pcl::PointIndices> cluster_indices;
    ec.setClusterTolerance (cluster_thr);
    ec.setMinClusterSize (15);
    ec.setMaxClusterSize (std::numeric_limits<int>::max());
    ec.setSearchMethod(tree);
    ec.setInputCloud (cloud);
    ec.extract (cluster_indices);
    pcl::ExtractIndices<PointNormal> extract;
    extract.setInputCloud (cloud);
    pcl::PointIndices::Ptr c_ind (new pcl::PointIndices);
    for (size_t i = 0; i < cluster_indices.size(); i++) {
        c_ind->indices.insert(std::end(c_ind->indices), std::begin(cluster_indices.at(i).indices), std::end(cluster_indices.at(i).indices));
    }
    extract.setIndices (c_ind);
    extract.setKeepOrganized(true);
    extract.setNegative (false);
    extract.filter (*cloud);

    return cluster_indices;
}
