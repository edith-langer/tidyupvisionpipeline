#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <limits>

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

#include "plane_object_extraction.h"


ExtractObjectsFromPlanes::ExtractObjectsFromPlanes(pcl::PointCloud<PointNormal>::Ptr cloud, Eigen::Vector4f main_plane_coeffs,
                                                   pcl::PointCloud<pcl::PointXYZ>::Ptr convex_hull_pts,
                                                   std::string output_path) {
    orig_cloud_ = cloud;
    main_plane_coeffs_ = main_plane_coeffs;
    convex_hull_pts_ = convex_hull_pts;
    result_path_ = output_path;
}

ExtractObjectsFromPlanes::~ExtractObjectsFromPlanes() {}

void ExtractObjectsFromPlanes::setResultPath(std::string path) {
    result_path_ = path;
}

std::vector<PlaneWithObjInd> ExtractObjectsFromPlanes::computeObjectsOnPlanes(pcl::PointCloud<PointNormal>::Ptr checked_plane_points_cloud) {
    std::string plane_result_path = result_path_ + "/plane_results/";
    boost::filesystem::create_directories(plane_result_path);
    std::cout << "Extract objects supported by a plane " << std::endl;
    std::vector<PlaneWithObjInd> plane_objects = extractObjectInd(checked_plane_points_cloud);

    return plane_objects;
}

std::vector<PlaneWithObjInd> ExtractObjectsFromPlanes::extractObjectInd(pcl::PointCloud<PointNormal>::Ptr checked_plane_points_cloud) {

    //plane coeffs from voxblox with low resolution are probably not good enough to extract the plane
    //crop the input cloud and run ransac to detect it
    //we get convex hull points for each plane from the DB, can we use cropHull from pcl?
    // get min and max point from convex hull (should be 2D), crop in x and y direction with some margin, run ransac again
    std::vector<PlaneWithObjInd> plane_object_result;

    PointNormal nan_point;
    nan_point.x = nan_point.y = nan_point.z = std::numeric_limits<float>::quiet_NaN();

    //---------------------------setup the main plane------------------------------------------------
    //crop cloud according to the convex hull points, find min and max values in x and y direction
    pcl::PointXYZ max_hull_pt, min_hull_pt;
    pcl::getMinMax3D(*convex_hull_pts_, min_hull_pt, max_hull_pt);

    //add some margin because hull points were computed from a full room reconstruction that may include drift
    float margin = 0.5;
    pcl::PointCloud<PointNormal>::Ptr cropped_cloud(new pcl::PointCloud<PointNormal>);
    pcl::PointCloud<PointNormal>::Ptr cropped_cloud_for_plane(new pcl::PointCloud<PointNormal>);
    pcl::PassThrough<PointNormal> pass;
    pass.setInputCloud(orig_cloud_);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(min_hull_pt.x - margin, max_hull_pt.x + margin);
    pass.setKeepOrganized(true);
    pass.filter(*cropped_cloud);
    pass.setInputCloud(cropped_cloud);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(min_hull_pt.y - margin, max_hull_pt.y + margin);
    pass.filter(*cropped_cloud);
    pass.setInputCloud(cropped_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(min_hull_pt.z - 0.1, max_hull_pt.z + 0.1);
    pass.filter(*cropped_cloud_for_plane); //we want to restrict the search space for the plane, but cropped_cloud is used later for extracting the objects and would cut them off

    if (cropped_cloud_for_plane->empty()) {
        return {}; //return empty object
    }
    pcl::io::savePCDFileBinary(result_path_ + "/cropped_cloud.pcd", *cropped_cloud_for_plane);

    //filter normals
    pcl::PointCloud<PointNormal>::Ptr cropped_cloud_filtered(new pcl::PointCloud<PointNormal>);
    pcl::ConditionAnd<PointNormal>::Ptr range_cond (new pcl::ConditionAnd<PointNormal> ());
    range_cond->addComparison (pcl::FieldComparison<PointNormal>::ConstPtr (new pcl::FieldComparison<PointNormal> ("normal_x", pcl::ComparisonOps::GT, -0.25))); //~15 deg
    range_cond->addComparison (pcl::FieldComparison<PointNormal>::ConstPtr (new pcl::FieldComparison<PointNormal> ("normal_x", pcl::ComparisonOps::LT, 0.25)));
    range_cond->addComparison (pcl::FieldComparison<PointNormal>::ConstPtr (new pcl::FieldComparison<PointNormal> ("normal_y", pcl::ComparisonOps::GT, -0.25)));
    range_cond->addComparison (pcl::FieldComparison<PointNormal>::ConstPtr (new pcl::FieldComparison<PointNormal> ("normal_y", pcl::ComparisonOps::LT, 0.25)));
    // build the filter
    pcl::ConditionalRemoval<PointNormal> condrem;
    condrem.setCondition (range_cond);
    condrem.setInputCloud(cropped_cloud_for_plane);
    condrem.setKeepOrganized(true);
    condrem.filter (*cropped_cloud_filtered);


    if (cropped_cloud_filtered->size() == 0) {
        std::cerr << "Not enough points left after cropping the input plane cloud " << std::endl;
        return {}; //return empty object
    }

    //    pcl::PointCloud<PointNormal>::Ptr biggest_cluster_plane (new pcl::PointCloud<PointNormal>);
    //    pcl::ExtractIndices<PointNormal> extract_biggest_cluster;
    //    extract_biggest_cluster.setInputCloud (cropped_cloud_for_plane);
    //    extract_biggest_cluster.setIndices (boost::make_shared<pcl::PointIndices>(clusters_cropped_cloud[0])); //the cluster result should be sorted, first element is the biggest
    //    extract_biggest_cluster.setNegative (false);
    //    extract_biggest_cluster.setKeepOrganized(true);
    //    extract_biggest_cluster.filter(*biggest_cluster_plane);
    //    pcl::io::savePCDFileBinary(result_path_ + "/biggest_cluster.pcd", *biggest_cluster_plane);

    //set plane points to nan that were already used before
    if (!checked_plane_points_cloud->empty()) {
        pcl::PointCloud<PointNormal>::Ptr checked_plane_points_cloud_cropped(new pcl::PointCloud<PointNormal>);
        pass.setInputCloud(checked_plane_points_cloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(min_hull_pt.x - margin, max_hull_pt.x + margin);
        pass.setKeepOrganized(false);
        pass.filter(*checked_plane_points_cloud_cropped);
        pass.setInputCloud(checked_plane_points_cloud_cropped);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(min_hull_pt.y - margin, max_hull_pt.y + margin);
        pass.filter(*checked_plane_points_cloud_cropped);
        pass.setInputCloud(checked_plane_points_cloud_cropped);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(min_hull_pt.z - 0.1, max_hull_pt.z + 0.1);
        pass.filter(*checked_plane_points_cloud_cropped);
        if (!checked_plane_points_cloud_cropped->empty()) {
            pcl::KdTreeFLANN<PointNormal> kdtree;
            kdtree.setInputCloud (checked_plane_points_cloud_cropped);
            std::vector<int> pointIdxSearch;
            std::vector<float> pointSquaredDistance;
            for (size_t i = 0; i < cropped_cloud_filtered->size(); i++) {
                PointNormal &search_point = cropped_cloud_filtered->points[i];
                if (pcl::isFinite(search_point)) {
                    if ( kdtree.radiusSearch (search_point, 0.05, pointIdxSearch, pointSquaredDistance) > 0 )
                    {
                        if (pointIdxSearch.size() > 0) {   //point was already used as plane point
                            search_point = nan_point;
                        }
                    }
                }
            }
        }
        pcl::io::savePCDFileBinary(result_path_ + "/previously_used_plane_points.pcd", *checked_plane_points_cloud);
    }


    pcl::io::savePCDFileBinary(result_path_ + "/potential_plane.pcd", *cropped_cloud_filtered);

    std::cout << "potential plane size  " << cropped_cloud_filtered->size() << std::endl;
    std::vector<int> nan_ind_pot_plane;
    pcl::PointCloud<PointNormal>::Ptr pot_plane_no_nans (new pcl::PointCloud<PointNormal>);
    pcl::removeNaNFromPointCloud(*cropped_cloud_filtered, *pot_plane_no_nans, nan_ind_pot_plane);
    std::cout << "potential plane size  wo nans" << pot_plane_no_nans->size() << std::endl;

    //cluster the cropped cloud
    pcl::EuclideanClusterExtraction<PointNormal> ec_cropped_cloud;
    std::vector<pcl::PointIndices> clusters_cropped_cloud;
    ec_cropped_cloud.setClusterTolerance (0.05);
    ec_cropped_cloud.setMinClusterSize (50);
    ec_cropped_cloud.setMaxClusterSize (std::numeric_limits<int>::max());
    ec_cropped_cloud.setInputCloud (pot_plane_no_nans);
    ec_cropped_cloud.extract (clusters_cropped_cloud);

    if (clusters_cropped_cloud.size() == 0) {
        std::cerr << "Not enough points left after cropping the input plane cloud " << std::endl;
        return {}; //return empty object
    }

    std::cout << "clustered planes into " << clusters_cropped_cloud.size() << " planes" << std::endl;

    //transform back to original indices
    for (pcl::PointIndices &ind : clusters_cropped_cloud) {
        for (size_t i = 0; i < ind.indices.size(); i++) {
                ind.indices[i] = nan_ind_pot_plane[ind.indices[i]];
        }
    }

    for (size_t c = 0; c < clusters_cropped_cloud.size(); c++) { //the size should be quite small, usually 1 or 2 planes
        pcl::ExtractIndices<PointNormal> extract_plane_cluster;
        extract_plane_cluster.setInputCloud (cropped_cloud);
        extract_plane_cluster.setIndices (boost::make_shared<pcl::PointIndices>(clusters_cropped_cloud[c])); //the cluster result should be sorted, first element is the biggest
        extract_plane_cluster.setNegative (false);
        extract_plane_cluster.setKeepOrganized(true);
        extract_plane_cluster.filter(*cropped_cloud_filtered);

        //remove NANs before applying RANSAC
        std::vector<int> nan_ind;
        pcl::PointCloud<PointNormal>::Ptr rough_plane_no_nans_cloud (new pcl::PointCloud<PointNormal>);
        pcl::removeNaNFromPointCloud(*cropped_cloud_filtered, *rough_plane_no_nans_cloud, nan_ind);

        // Create the segmentation object
        pcl::SACSegmentation<PointNormal> seg;
        seg.setOptimizeCoefficients (true);
        seg.setModelType (pcl::SACMODEL_PERPENDICULAR_PLANE);
        seg.setMaxIterations(500);
        seg.setMethodType (pcl::SAC_RANSAC);
        seg.setDistanceThreshold (0.01);
        seg.setAxis(Eigen::Vector3f(0,0,1));
        seg.setEpsAngle(5.0f * (M_PI/180.0f)); //without setting an angle, the axis is ignored and all planes get segmented
        pcl::ModelCoefficients::Ptr new_plane_coefficients (new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr main_plane_inliers (new pcl::PointIndices ());
        seg.setInputCloud(rough_plane_no_nans_cloud);
        seg.segment (*main_plane_inliers, *new_plane_coefficients);

        std::cout << "Extracted main plane with " << main_plane_inliers->indices.size() << " inliers" << std::endl;

        //convert main_plane_inlierst back to original indices
        for (size_t i = 0; i < main_plane_inliers->indices.size(); i++) {
            main_plane_inliers->indices[i] = nan_ind[main_plane_inliers->indices[i]];
        }


        PlaneStruct main_plane;
        main_plane.plane_ind = main_plane_inliers;
        main_plane.coeffs = new_plane_coefficients;

        for (size_t i = 0; i < main_plane_inliers->indices.size(); i++) {
            main_plane.avg_z += cropped_cloud->points[main_plane_inliers->indices[i]].z;
        }
        main_plane.avg_z /= main_plane_inliers->indices.size();

        //-----------------------------------------------------------------------------------------------


        pcl::PointCloud<PointNormal>::Ptr plane_cloud(new pcl::PointCloud<PointNormal>);
        pcl::PointCloud<PointNormal>::Ptr remaining_cloud(new pcl::PointCloud<PointNormal>);

        //Extract plane
        pcl::ExtractIndices<PointNormal> extract;
        extract.setInputCloud (cropped_cloud);
        extract.setIndices (main_plane.plane_ind);
        extract.setNegative (false);
        extract.setKeepOrganized(true);
        extract.filter (*plane_cloud);
        pcl::io::savePCDFileBinary(result_path_ + "/extracted_plane" + std::to_string(c) + ".pcd", *plane_cloud);

        extract.setInputCloud (cropped_cloud);
        extract.setIndices (main_plane.plane_ind);
        extract.setNegative (true);
        extract.setKeepOrganized(true);
        extract.filter (*remaining_cloud);
        if (remaining_cloud->empty())
            continue; //return empty object
        pcl::io::savePCDFileBinary(result_path_ + "/remaining_points" + std::to_string(c) + ".pcd", *remaining_cloud);


        //check if the plane is intersecting with the original convex hull of the plane
        if (!intersectCHWithPlane(convex_hull_pts_, plane_cloud))
            continue;


        //Check if the plane is a good plane by removing normals not pointing in the z-direction
        // build the condition
        pcl::PointCloud<PointNormal>::Ptr cloud_plane_filtered(new pcl::PointCloud<PointNormal>);
        condrem.setCondition (range_cond);
        condrem.setInputCloud(plane_cloud);
        condrem.setKeepOrganized(true);
        condrem.filter (*cloud_plane_filtered);

        //remove nans before checking size
        pcl::PointCloud<PointNormal>::Ptr plane_filtered_wo_nans_cloud(new pcl::PointCloud<PointNormal>);
        pcl::removeNaNFromPointCloud(*cloud_plane_filtered, *plane_filtered_wo_nans_cloud, nan_ind);
        if (plane_filtered_wo_nans_cloud->size () < 30)
        {
            std::cerr << "Not enough points left after filtering the plane for wrong normals (< 30) " << std::endl;
            continue; //return empty object
        }
        pcl::io::savePCDFileBinary(result_path_ + "/normals_filtered" + std::to_string(c) + ".pcd", *cloud_plane_filtered);

        //the filtered normal cloud can have some spourious points left, therefore we remove very small clusters
        pcl::EuclideanClusterExtraction<PointNormal> ec_normal;
        std::vector<pcl::PointIndices> cluster_indices_normals;
        ec_normal.setClusterTolerance (0.05);
        ec_normal.setMinClusterSize (5);
        ec_normal.setMaxClusterSize (std::numeric_limits<int>::max());
        ec_normal.setInputCloud (plane_filtered_wo_nans_cloud);
        ec_normal.extract (cluster_indices_normals);

        //transform back to original indices
        for (pcl::PointIndices &ind : cluster_indices_normals) {
            for (size_t i = 0; i < ind.indices.size(); i++) {
                ind.indices[i] = nan_ind[ind.indices[i]];
            }
        }

        //this code should not be needed. it merges separated planes again in one object, but we are looking at one plane only anyway
        pcl::PointIndices::Ptr c_ind (new pcl::PointIndices);
        for (size_t i = 0; i < cluster_indices_normals.size(); i++) {
            if ((cluster_indices_normals.at(i).indices.size()) > 300)
                c_ind->indices.insert(std::end(c_ind->indices), std::begin(cluster_indices_normals.at(i).indices), std::end(cluster_indices_normals.at(i).indices));
        }
        pcl::ExtractIndices<PointNormal> extract_normal;
        pcl::PointCloud<PointNormal>::Ptr cloud_plane_filtered_notOrganized(new pcl::PointCloud<PointNormal>);
        extract_normal.setInputCloud(cloud_plane_filtered);
        extract_normal.setIndices (c_ind);
        extract_normal.setNegative (false);
        extract_normal.filter (*cloud_plane_filtered_notOrganized);
        if (cloud_plane_filtered_notOrganized->size()< 30) {
            std::cerr << "Not enough points left after filtering the plane for small cluster (< 30) " << std::endl;
            continue; //return empty object
        }
        extract_normal.setKeepOrganized(true);
        extract_normal.filter(*cloud_plane_filtered);

        //set the plane inliers to the filtered outcome
        //update z-value
        pcl::PointIndices::Ptr filtered_plane_ind(new pcl::PointIndices);
        float new_z_value =0.0f;
        // plane_ind are not from the original cloud, but plane_cloud!
        for (size_t i = 0; i < c_ind->indices.size(); i++) {
            int ind =  c_ind->indices[i];
            new_z_value+=plane_cloud->points.at(ind).z;
            filtered_plane_ind->indices.push_back(ind);
        }
        main_plane.avg_z = new_z_value/c_ind->indices.size();
        main_plane.plane_ind =filtered_plane_ind;

        pcl::io::savePCDFileBinary(result_path_ + "/normals_filtered_clustered" + std::to_string(c) + ".pcd", *cloud_plane_filtered);


        //compute convex hull
        std::cout << "Input to concave hull computation has " << cloud_plane_filtered_notOrganized->size() << " points" << std::endl;
        // Create a Convex Hull representation of the projected inliers (cloud must not have NANs!)
        pcl::PointCloud<PointNormal>::Ptr cloud_hull (new pcl::PointCloud<PointNormal>);
        pcl::ConvexHull<PointNormal> chull;
        chull.setInputCloud (cloud_plane_filtered_notOrganized);
        chull.setDimension(2);
        //chull.setAlpha(50);
        chull.reconstruct (*cloud_hull);
        std::cout << "Convex hull has: " << cloud_hull->points.size () << " data points." << std::endl;
        pcl::io::savePCDFileBinary(result_path_ + "/convex_hull" + std::to_string(c) + ".pcd", *cloud_hull);

        shrinkConvexHull(cloud_hull, 0.02);
        pcl::io::savePCDFileBinary(result_path_ + "/convex_hull_shrinked" + std::to_string(c) + ".pcd", *cloud_hull);

        // segment those points that are in the polygonal prism
        pcl::ExtractPolygonalPrismData<PointNormal> ex_prism;
        ex_prism.setViewPoint(0,0,5); //this is used to flip the plane normal towards the viewpoint, if necessary
        ex_prism.setInputCloud (cropped_cloud);
        ex_prism.setInputPlanarHull (cloud_hull);
        ex_prism.setHeightLimits(0.01, 0.3);
        pcl::PointIndices::Ptr object_indices (new pcl::PointIndices);
        ex_prism.segment (*object_indices);

        if (object_indices->indices.size() > 0) {
            // Get and show all points retrieved by the hull.
            pcl::PointCloud<PointNormal>::Ptr objects(new pcl::PointCloud<PointNormal>);
            extract.setInputCloud(cropped_cloud);
            extract.setNegative(false);
            extract.setIndices(object_indices);
            extract.setKeepOrganized(true);
            extract.filter(*objects);
            pcl::io::savePCDFileBinary(result_path_ + "/objects" + std::to_string(c) + ".pcd", *objects);

            //remove plane points by region growing
            pcl::PointCloud<PointNormal>::Ptr plane_obj_combined (new pcl::PointCloud<PointNormal>(objects->width, objects->height, nan_point));
            for (size_t i = 0; i < object_indices->indices.size(); i++)
                plane_obj_combined->points[object_indices->indices[i]] = objects->points[object_indices->indices[i]];
            for (size_t i = 0; i < filtered_plane_ind->indices.size(); i++)
                plane_obj_combined->points[filtered_plane_ind->indices[i]] = cloud_plane_filtered->points[filtered_plane_ind->indices[i]];

            pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
            pcl::copyPointCloud(*plane_obj_combined, *scene_normals);
            RegionGrowing<PointNormal, PointNormal> region_growing(plane_obj_combined, cloud_plane_filtered, scene_normals, false, 20.0, 5);
            std::vector<int> all_plane_points = region_growing.compute();
            pcl::PointIndices::Ptr plane_ind (new pcl::PointIndices);
            plane_ind->indices = all_plane_points;
            extract.setInputCloud(plane_obj_combined);
            extract.setNegative(true);
            extract.setIndices(plane_ind);
            extract.setKeepOrganized(true);
            extract.filter(*objects);
            extract.setNegative(false);
            extract.filter(*plane_cloud);
            pcl::io::savePCDFileBinary(result_path_ + "/objects_wo_plane_pts" + std::to_string(c) + ".pcd", *objects);
            //remove all_plane_points that are in object_indices
            std::sort(all_plane_points.begin(), all_plane_points.end());
            std::sort(object_indices->indices.begin(), object_indices->indices.end());
            std::vector<int> diff;
            std::set_difference(object_indices->indices.begin(), object_indices->indices.end(), all_plane_points.begin(), all_plane_points.end(),
                                std::inserter(diff, diff.begin()));
            object_indices->indices = diff;

            //add plane cloud to the checked plane points
            *checked_plane_points_cloud += *plane_cloud;

            if (object_indices->indices.size() == 0) {
                std::cerr << "No objects found" << std::endl;
                continue;
            }

            //check if objects are flying
            std::cout << "Object indices size before filtering flying objects" << object_indices->indices.size();
            filter_flying_objects(objects, object_indices, plane_cloud);
            std::cout << " and after filtering: " << object_indices->indices.size() <<std::endl;

            //check if objects are planar
            std::cout << "Object indices size before filtering planar objects " << object_indices->indices.size();
            filter_planar_objects(objects, object_indices);
            std::cout << " and after filtering: " << object_indices->indices.size() <<std::endl;

            if (object_indices->indices.size() > 0) {

                pcl::PointCloud<PointNormal>::Ptr objects(new pcl::PointCloud<PointNormal>);
                extract.setInputCloud(orig_cloud_);
                extract.setNegative(false);
                extract.setIndices(object_indices);
                extract.setKeepOrganized(true);
                extract.filter(*objects);
                pcl::io::savePCDFileBinary(result_path_ + "/objects_filtered" + std::to_string(c) + ".pcd", *objects);

                pcl::PointCloud<PointNormal>::Ptr plane(new pcl::PointCloud<PointNormal>);
                extract.setInputCloud(orig_cloud_);
                extract.setNegative(false);
                extract.setIndices(plane_ind);
                extract.setKeepOrganized(true);
                extract.filter(*plane);
                pcl::io::savePCDFileBinary(result_path_ + "/plane" + std::to_string(c) + ".pcd", *plane);


                PlaneStruct supp_plane;
                supp_plane.plane_ind = boost::make_shared<pcl::PointIndices>(*(plane_ind));
                supp_plane.avg_z =  main_plane.avg_z;
                supp_plane.coeffs = boost::make_shared<pcl::ModelCoefficients>(*(main_plane.coeffs));

                PlaneWithObjInd result;
                result.obj_indices = object_indices->indices;
                result.plane = supp_plane;

                plane_object_result.push_back(result);

                //return plane_object_result;
            } else {
                std::cerr << "No objects found" << std::endl;
                //continue;
            }
        } else {
            std::cerr << "No objects found" << std::endl;
            //return PlaneWithObjInd();
        }
    }
    std::cout << "Done with extracting objects from planes" << std::endl;
    return plane_object_result;
}



void ExtractObjectsFromPlanes::filter_flying_objects(pcl::PointCloud<PointNormal>::Ptr cloud, pcl::PointIndices::Ptr ind, pcl::PointCloud<PointNormal>::Ptr plane) {
    std::vector<int> nan_ind;
    pcl::PointCloud<PointNormal>::Ptr cloud_no_nans(new pcl::PointCloud<PointNormal>);
    pcl::removeNaNFromPointCloud(*cloud, *cloud_no_nans, nan_ind);
    
    // Create EuclideanClusterExtraction and set parameters
    pcl::EuclideanClusterExtraction<PointNormal> ec;
    std::vector<pcl::PointIndices> cluster_indices;
    ec.setClusterTolerance (0.015);
    ec.setMinClusterSize (15);
    ec.setMaxClusterSize (std::numeric_limits<int>::max());
    ec.setInputCloud (cloud_no_nans);
    ec.extract (cluster_indices);

    //transform back to original indices
    for (pcl::PointIndices &ind : cluster_indices) {
        for (size_t i = 0; i < ind.indices.size(); i++) {
            ind.indices[i] = nan_ind[ind.indices[i]];
        }
    }


    std::vector<int> ind_to_be_removed;
    int i = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        pcl::PointCloud<PointNormal>::Ptr cloud_cluster (new pcl::PointCloud<PointNormal>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
            cloud_cluster->points.push_back (cloud->points[*pit]);
        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        PointNormal minPt, maxPt;
        pcl::getMinMax3D (*cloud_cluster, minPt, maxPt);

        //crop plane according to cluster dimensions
        pcl::PointCloud<PointNormal>::Ptr cropped_plane_cloud(new pcl::PointCloud<PointNormal>);
        pcl::PassThrough<PointNormal> pass;
        pass.setInputCloud(plane);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(minPt.x - 0.15, maxPt.x + 0.15);
        pass.setKeepOrganized(false);
        pass.filter(*cropped_plane_cloud);
        pass.setInputCloud(cropped_plane_cloud);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(minPt.y - 0.15, maxPt.y + 0.15);
        pass.filter(*cropped_plane_cloud);

        float avg_cropped_plane_z = 0;
        for (size_t p = 0; p < cropped_plane_cloud->size(); p++) {
            avg_cropped_plane_z += cropped_plane_cloud->points[p].z;
        }
        avg_cropped_plane_z /= cropped_plane_cloud->size();

        if (minPt.z - avg_cropped_plane_z > 0.05) {
            std::cerr << "Cluster is flying. We remove it." << std::endl;
            ind_to_be_removed.insert(std::end(ind_to_be_removed), std::begin(cluster_indices.at(i).indices), std::end(cluster_indices.at(i).indices));
        }
        i++;
    }
    ind->indices.erase(std::remove_if(std::begin(ind->indices),std::end(ind->indices),
                                      [&](int x){return find(std::begin(ind_to_be_removed),std::end(ind_to_be_removed),x)!=std::end(ind_to_be_removed);}), std::end(ind->indices) );
}

void ExtractObjectsFromPlanes::filter_planar_objects(pcl::PointCloud<PointNormal>::Ptr cloud, pcl::PointIndices::Ptr ind) {

    std::vector<int> nan_ind;
    pcl::PointCloud<PointNormal>::Ptr cloud_no_nans(new pcl::PointCloud<PointNormal>);
    pcl::removeNaNFromPointCloud(*cloud, *cloud_no_nans, nan_ind);

    // Create EuclideanClusterExtraction and set parameters
    pcl::EuclideanClusterExtraction<PointNormal> ec;
    std::vector<pcl::PointIndices> cluster_indices;
    ec.setClusterTolerance (0.015);
    ec.setMinClusterSize (15);
    ec.setMaxClusterSize (std::numeric_limits<int>::max());
    ec.setInputCloud (cloud_no_nans);
    ec.extract (cluster_indices);

    //transform back to original indices
    for (pcl::PointIndices &ind : cluster_indices) {
        for (size_t i = 0; i < ind.indices.size(); i++) {
            ind.indices[i] = nan_ind[ind.indices[i]];
        }
    }

    std::vector<int> ind_to_be_removed;
    int i = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        pcl::PointCloud<PointNormal>::Ptr cloud_cluster (new pcl::PointCloud<PointNormal>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
            cloud_cluster->points.push_back (cloud->points[*pit]);
        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        if (isObjectPlanar(cloud_cluster, 0.01, 0.9)) {
            std::cerr << "Cluster is a plane. We remove it." << std::endl;
            ind_to_be_removed.insert(std::end(ind_to_be_removed), std::begin(cluster_indices.at(i).indices), std::end(cluster_indices.at(i).indices));
        }
        i++;
    }
    ind->indices.erase(std::remove_if(std::begin(ind->indices),std::end(ind->indices),
                                      [&](int x){return find(std::begin(ind_to_be_removed),std::end(ind_to_be_removed),x)!=std::end(ind_to_be_removed);}), std::end(ind->indices) );
}


void ExtractObjectsFromPlanes::shrinkConvexHull(pcl::PointCloud<PointNormal>::Ptr hull_cloud, float distance) {
    Eigen::Vector4f centroid4f; //last element is 1
    pcl::compute3DCentroid(*hull_cloud, centroid4f);
    Eigen::Vector3f centroid3f; centroid3f << centroid4f[0], centroid4f[1], centroid4f[2];

    for (size_t i = 0; i < hull_cloud->size(); i++) {
        Eigen::Vector3f h_pt_vec = hull_cloud->points[i].getVector3fMap();
        Eigen::Vector3f subtract = centroid3f - h_pt_vec;
        subtract.normalize();

        //move the point a little to the center
        hull_cloud->points[i].x = hull_cloud->points[i].x + subtract[0]*distance;
        hull_cloud->points[i].y = hull_cloud->points[i].y + subtract[1]*distance;
        hull_cloud->points[i].z = hull_cloud->points[i].z + subtract[2]*distance;
    }
}

bool ExtractObjectsFromPlanes::intersectCHWithPlane(pcl::PointCloud<PointXYZ>::ConstPtr cloud_hull_orig, pcl::PointCloud<PointNormal>::ConstPtr plane_cloud) {
    pcl::PointCloud<PointXYZ>::Ptr plane_cloud_xyz (new pcl::PointCloud<PointXYZ>);
    pcl::copyPointCloud(*plane_cloud, *plane_cloud_xyz);

    std::vector<int> nan_ind;
    pcl::removeNaNFromPointCloud(*plane_cloud_xyz, *plane_cloud_xyz, nan_ind);

    pcl::PointCloud<PointXYZ>::Ptr cloud_hull (new pcl::PointCloud<PointXYZ>);
    std::vector<pcl::Vertices> polygons;
    pcl::ConvexHull<PointXYZ> chull;
    chull.setInputCloud (cloud_hull_orig);
    chull.setDimension(2);
    chull.reconstruct (*cloud_hull, polygons);

    pcl::PointCloud<PointXYZ>::Ptr intersection_cloud (new pcl::PointCloud<PointXYZ>);
    pcl::CropHull<PointXYZ> crop_filter;
    crop_filter.setInputCloud (plane_cloud_xyz);
    crop_filter.setHullCloud (cloud_hull);
    crop_filter.setHullIndices (polygons);
    crop_filter.setDim (2);
    crop_filter.filter (*intersection_cloud);

    if (intersection_cloud->empty()) {
        return false;
    }
    return true;
}
