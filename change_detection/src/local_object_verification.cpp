#include "local_object_verification.h"

LocalObjectVerification::LocalObjectVerification(std::vector<PlaneWithObjInd> potential_objects, pcl::PointCloud<PointType>::Ptr ref_cloud,
                                                 pcl::PointCloud<PointType>::Ptr curr_cloud, LocalObjectVerificationParams params)
{
    this->potential_objects = potential_objects;
    this->ref_cloud = ref_cloud;
    this->curr_cloud = curr_cloud;

    params_ = params;

    this->debug_output_path="";
}

void LocalObjectVerification::setDebugOutputPath (std::string debug_output_path) {
    this->debug_output_path = debug_output_path;
}

std::vector<PlaneWithObjInd> LocalObjectVerification::verify_changed_objects() {
    std::vector<PlaneWithObjInd> verified_objects;

    std::string crop_result_path = debug_output_path + "/crop_results/";
    boost::filesystem::create_directories(crop_result_path);

    //for each object cluster crop the scene and align only the cropped part
    for (size_t c = 0; c < potential_objects.size(); c++) {
        pcl::PointCloud<PointType>::Ptr cloud_cluster(new pcl::PointCloud<PointType>);
        for (size_t p =0; p < potential_objects.at(c).obj_indices.size(); p++) {
            int ind = potential_objects.at(c).obj_indices.at(p);
            if (pcl::isFinite(curr_cloud->points.at(ind))) {
                cloud_cluster->points.push_back(curr_cloud->points.at(ind));
            }
        }
        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        std::cout << "Cluster " << c << " has " << cloud_cluster->size() << " points." << std::endl;
        pcl::io::savePCDFileBinary(crop_result_path + "/cluster" + std::to_string(c) +".pcd", *cloud_cluster);

        //use that plane to align the cluster in z-direction
        pcl::ModelCoefficients::Ptr supp_plane_coeff = potential_objects[c].plane.coeffs;
        pcl::PointCloud<PointType>::Ptr supporting_plane(new pcl::PointCloud<PointType>);
        pcl::ExtractIndices<PointType> extract;
        extract.setInputCloud (curr_cloud);
        extract.setIndices(potential_objects[c].plane.plane_ind);
        extract.setKeepOrganized(true);
        extract.setNegative (false);
        extract.filter(*supporting_plane);
        pcl::io::savePCDFileBinary(crop_result_path + "/supp_plane_" + std::to_string(c) +".pcd", *supporting_plane);

        //crop the supporting plane to improve ICP speed
        //get min and max values
        PointType minPt_object, maxPt_object;
        pcl::getMinMax3D (*cloud_cluster, minPt_object, maxPt_object);

        //crop cloud
        float object_plane_margin = 0.1;
        pcl::PointCloud<PointType>::Ptr crop(new pcl::PointCloud<PointType>);
        pcl::PassThrough<PointType> pass_plane_object;
        pass_plane_object.setInputCloud(supporting_plane);
        pass_plane_object.setFilterFieldName("x");
        pass_plane_object.setFilterLimits(minPt_object.x - object_plane_margin, maxPt_object.x + object_plane_margin);
        pass_plane_object.setKeepOrganized(false);
        pass_plane_object.filter(*supporting_plane);
        pass_plane_object.setInputCloud(supporting_plane);
        pass_plane_object.setFilterFieldName("y");
        pass_plane_object.setFilterLimits(minPt_object.y - object_plane_margin, maxPt_object.y + object_plane_margin);
        pass_plane_object.filter(*supporting_plane);


        //this should not happen
        if (supporting_plane->size() == 0) { //no plane found --> flying object --> remove it
            //                        for (size_t i = 0; i < object_clusters.at(c).indices.size(); i++) {
            //                            PointType &p = semSeg_object_cloud->points.at(object_clusters.at(c).indices.at(i));
            //                            p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
            //                        }
            std::cout << "Cluster " << std::to_string(c) << " gets removed because no supporting plane found." << std::endl;
            continue;
        }
        pcl::io::savePCDFileBinary(crop_result_path + "/cropped_supp_plane_" + std::to_string(c) +".pcd", *supporting_plane);


        //crop reference cloud
        pcl::PointCloud<PointType>::Ptr static_crop_plane(new pcl::PointCloud<PointType>);
        pcl::PointCloud<PointType>::Ptr static_crop_cluster(new pcl::PointCloud<PointType>);
        pcl::PassThrough<PointType> pass;
        pass.setInputCloud(ref_cloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(minPt_object.x - params_.add_crop_static, maxPt_object.x + params_.add_crop_static);
        pass.setKeepOrganized(true);
        pass.filter(*static_crop_plane);
        pass.setInputCloud(static_crop_plane);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(minPt_object.y - params_.add_crop_static, maxPt_object.y + params_.add_crop_static);
        pass.setKeepOrganized(true);
        pass.filter(*static_crop_plane);
        pass.setInputCloud(static_crop_plane);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(minPt_object.z - params_.add_crop_static, maxPt_object.z + params_.add_crop_static);
        pass.setKeepOrganized(true);
        pass.filter(*static_crop_cluster);
        pass.setInputCloud(static_crop_cluster);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(minPt_object.z - params_.add_crop_static, minPt_object.z + params_.add_crop_static);
        pass.setKeepOrganized(true);
        pass.filter(*static_crop_plane);
        //static_crop_plane and static_crop_cluster are the same, possible to change the amount added in z-direction for the plane
        pcl::io::savePCDFileBinary(crop_result_path + "/static_crop_plane_"  + std::to_string(c) +".pcd", *static_crop_plane);
        pcl::io::savePCDFileBinary(crop_result_path + "/static_crop_cluster_" + std::to_string(c) +".pcd", *static_crop_cluster);

        //filter nans for ICP
        pcl::PointCloud<PointType>::Ptr static_crop_filtered(new pcl::PointCloud<PointType>);
        std::vector<int> ind;
        pcl::removeNaNFromPointCloud(*static_crop_plane, *static_crop_plane, ind);
        pcl::removeNaNFromPointCloud(*static_crop_cluster, *static_crop_filtered, ind);
        pcl::removeNaNFromPointCloud(*supporting_plane, * supporting_plane, ind);

        //Add the cluster to the result or don't add it. That's the question.
        if (static_crop_plane->size() == 0 || static_crop_filtered->size() == 0) {
            std::cout << "Nothing in the static crop (plane or cluster) " << c << ". Add the cluster to the result." << std::endl;
            PlaneWithObjInd res_object;
            res_object.plane = potential_objects[c].plane;
            res_object.obj_indices = potential_objects[c].obj_indices;
            verified_objects.push_back(res_object);
            continue;
        }
        //If there is no supporting plane for the cluster, we don't add it to the result
        if (supporting_plane->size() == 0) {
            std::cout << "Supporting plane contains only NANs for cluster " << c << ". Don't add cluster to result." << std::endl;
            continue;
        }

        //apply ICP to align the plane
        pcl::registration::WarpPointRigid4D<PointType, PointType>::Ptr warp_fcn_4d (new pcl::registration::WarpPointRigid4D<PointType, PointType>);

        // Create a TransformationEstimationLM object, and set the warp to it
        pcl::registration::TransformationEstimationLM<PointType, PointType>::Ptr te (new pcl::registration::TransformationEstimationLM<PointType, PointType>);
        te->setWarpFunction (warp_fcn_4d);

        //Align the supporting plane with the plane found in the reference scene
        std::cout << "Set up plane ICP with supp. plane of size " << supporting_plane->size() << " and static crop of size " << static_crop_filtered->size() << std::endl;
        pcl::PointCloud<PointType>::Ptr plane_registered(new pcl::PointCloud<PointType>());
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setTransformationEstimation(te);
        icp.setInputSource(supporting_plane);
        icp.setInputTarget(static_crop_plane);
        icp.setMaxCorrespondenceDistance(params_.icp_max_corr_dist_plane);
        icp.setRANSACOutlierRejectionThreshold(params_.icp_ransac_thr);
        icp.setMaximumIterations(params_.icp_max_iter);
        icp.setTransformationEpsilon (1e-9);
        icp.setTransformationRotationEpsilon(1 - 1e-15); //epsilon is the cos(angle)
        icp.align(*plane_registered);
        std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;

        if (icp.hasConverged() && icp.getFitnessScore()) {
            std::cout << "align cluster" << std::endl;
            //found transformation, but we want to save the cloud including nans
            Eigen::Matrix4f icp_trans_mat = icp.getFinalTransformation();
            pcl::transformPointCloudWithNormals(*cloud_cluster, *cloud_cluster, icp_trans_mat);
            pcl::transformPointCloudWithNormals(*supporting_plane, *supporting_plane, icp_trans_mat);
            pcl::io::savePCDFileBinary(crop_result_path + "/cluster_after_plane_icp_"  + std::to_string(c) +".pcd", *cloud_cluster);
            pcl::io::savePCDFileBinary(crop_result_path + "/supporting_plane_after_plane_icp_"  + std::to_string(c) +".pcd", *supporting_plane);

            //transform the supporting plane coeffs to be able to remove the plane in the static crop
            //p' = transpose(inverse(M))*p
            Eigen::Matrix4f inv_trans_mat = icp_trans_mat.inverse();
            Eigen::Matrix4f transposed_mat = inv_trans_mat.transpose();
            Eigen::Vector4f plane_coeffs(4);
            plane_coeffs << supp_plane_coeff->values[0], supp_plane_coeff->values[1], supp_plane_coeff->values[2], supp_plane_coeff->values[3];
            Eigen::Vector4f transformed_coeffs = transposed_mat * plane_coeffs;
            std::cout << "Transformed plane coefficients" << std::endl;
            std::cout << transformed_coeffs[0] << transformed_coeffs[1] << transformed_coeffs[2] << transformed_coeffs[3] << std::endl;

            //remove plane from static crop
            pcl::SampleConsensusModelPlane<PointType>::Ptr dit (new pcl::SampleConsensusModelPlane<PointType> (static_crop_filtered));
            std::vector<int> inliers;
            dit -> selectWithinDistance (transformed_coeffs, 0.01, inliers);
            std::cout << "Static crop plane inliers " << inliers.size() << std::endl;
            pcl::PointIndices::Ptr trans_plane_ind(new pcl::PointIndices());
            trans_plane_ind->indices = inliers;
            pcl::ExtractIndices<PointType> extract;
            extract.setInputCloud (static_crop_filtered);
            extract.setIndices (trans_plane_ind);
            extract.setNegative (true);
            extract.filter(*static_crop_filtered);
            if (static_crop_filtered->size() == 0) {
                std::cout << "Static crop is empty after removing plane. Add the cluster to the result cloud."<< std::endl;
                PlaneWithObjInd res_object;
                res_object.plane = potential_objects[c].plane;
                res_object.obj_indices = potential_objects[c].obj_indices;
                verified_objects.push_back(res_object);
                continue;
            }
            pcl::io::savePCDFileBinary(crop_result_path + "/static_crop_no_supp_plane_cluster"  + std::to_string(c) +".pcd", *static_crop_filtered);

            //now do a 3D transformation with the cluster and the static crop
            pcl::registration::WarpPointRigid3D<PointType, PointType>::Ptr warp_fcn_3d (new pcl::registration::WarpPointRigid3D<PointType, PointType>);
            te->setWarpFunction (warp_fcn_3d);

            pcl::PointCloud<PointType>::Ptr cluster_registered(new pcl::PointCloud<PointType>());
            pcl::IterativeClosestPointWithNormals<PointType, PointType> icp;
            icp.setTransformationEstimation(te);
            icp.setInputSource(cloud_cluster);
            icp.setInputTarget(static_crop_filtered);
            icp.setMaxCorrespondenceDistance(params_.icp_max_corr_dist);
            icp.setRANSACOutlierRejectionThreshold(params_.icp_ransac_thr);
            icp.setMaximumIterations(params_.icp_max_iter);
            icp.setTransformationEpsilon (1e-9);
            icp.setTransformationRotationEpsilon(1 - 1e-15); //epsilon is the cos(angle)
            icp.align(*cluster_registered);
            std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;

            //      if (icp.hasConverged() && icp.getFitnessScore()) {

            //transform cloud cluster
            icp_trans_mat = icp.getFinalTransformation();
            pcl::transformPointCloudWithNormals(*cloud_cluster, *cloud_cluster, icp_trans_mat);
            pcl::io::savePCDFileBinary(crop_result_path + "/cluster_after_final_icp_cluster_"  + std::to_string(c) +".pcd", *cloud_cluster);
            //compare current_crop with static_crop
            SceneDifferencingPoints scene_diff = SceneDifferencingPoints(params_.diff_dist);
            std::vector<int> diff_ind;
            std::vector<int> corresponding_ind;
            //scene_diff.setDistThreshold(sqrt(2)*0.01);//sqrt(2)*0.01/2);
            SceneDifferencingPoints::Cloud::Ptr diff = scene_diff.computeDifference(cloud_cluster, static_crop_cluster, diff_ind, corresponding_ind);
            if (diff_ind.size() != 0) { //if(diff_ind.size() == 0) --> perfect alignment
                pcl::io::savePCDFileBinary(crop_result_path + "/cloud_diff_cluster_" + std::to_string(c) +".pcd", *diff);
                std::cout << "Points that are different: " << diff->size() << std::endl;
            }

            //add points to the result cloud that have little overlap with the static scene
            std::vector<int> &cloud_indices = potential_objects[c].obj_indices;
            std::vector<int> v(cloud_indices.size());
            std::vector<int>::iterator it;
            it=std::set_difference (cloud_indices.begin(), cloud_indices.end(), diff_ind.begin(), diff_ind.end(), v.begin());
            v.resize(it-v.begin()); //v contains now the cluster_indices that were not in the diff_ind and are therefore static object points
            float overlap = (float)v.size()/cloud_indices.size();
            std::cout << "Cluster " << std::to_string(c) << " overlap of "  << std::to_string(overlap) << std::endl;

            //TODO common hypothesis verification together with PPF?
            //----------------------------linear combination of color and overlap---------------
            if (corresponding_ind.size() > 0 ) { //check color simialrity only if there were overlapping points in the static cluster
                // Copy all the data fields from the input cloud to the output one
                pcl::PointCloud<PointType>::Ptr static_overlap_cloud(new pcl::PointCloud<PointType>);
                pcl::copyPointCloud(*ref_cloud, corresponding_ind, *static_overlap_cloud);
                pcl::io::savePCDFileBinary(crop_result_path + "/static_overlap_cluster" + std::to_string(c) +".pcd", *static_overlap_cloud);

                ColorHistogram hist;
                double color_corr = hist.colorCorr(static_overlap_cloud, cloud_cluster);
                std::cout << "Color correlation: " << color_corr << " for Cluster " << std::to_string(c) << std::endl;

                double score = params_.color_weight*color_corr + params_.overlap_weight*overlap;
                if (score < params_.min_score_thr) {
                    PlaneWithObjInd res_object;
                    res_object.plane = potential_objects[c].plane;
                    res_object.obj_indices = cloud_indices;
                    verified_objects.push_back(res_object);
                    std::cout << "Cluster " << std::to_string(c) << " gets added. Score: " << score << std::endl;
                }
            } else {
                std::cout << "No corresponding indices in the reference cloud" << std::endl;
                PlaneWithObjInd res_object;
                res_object.plane = potential_objects[c].plane;
                res_object.obj_indices = cloud_indices;
                verified_objects.push_back(res_object);
                std::cout << "Cluster " << std::to_string(c) << " gets added." << std::endl;
            }
            //------------------------------------------------------------------------------------
            //                if (overlap < overlap_thr) { //if less than x percent of the cluster are in the diff, add the whole cluster
            //                        PlaneWithObjInd res_object;
            //                        res_object.plane = plane_object.plane;
            //                        res_object.obj_indices = cloud_indices;
            //                        verified_objects.push_back(res_object);
            //                        std::cout << "Plane " << po << "; Cluster " << std::to_string(c) << " gets added." << std::endl;
            //                } else { //high overlap, check color similarity
            //                    double color_corr = 0;
            //                    //check if there is some correlation in color
            //                    if (corresponding_ind.size() > 0 && overlap < 0.9) { //check color simialrity only if the overlap is not higher than 0.9 and if there were overlapping points in the static cluster
            //                        // Copy all the data fields from the input cloud to the output one
            //                        pcl::PointCloud<PointType>::Ptr static_overlap_cloud(new pcl::PointCloud<PointType>);
            //                        pcl::copyPointCloud(*ref_cloud, corresponding_ind, *static_overlap_cloud);
            //                        pcl::io::savePCDFileBinary(crop_result_path + "/static_overlap" + std::to_string(po) + "_cluster" + std::to_string(c) +".pcd", *static_overlap_cloud);
            //                        //WARNING: cloud_cluster saves original ind in its rgb field!!!
            //                        pcl::PointCloud<PointType>::Ptr cloud_cluster_orig(new pcl::PointCloud<PointType>);
            //                        for (size_t i = 0; i < cloud_cluster->size(); i++) {
            //                            cloud_cluster_orig->points.push_back(curr_cloud->points[cloud_cluster->points[i].rgb]);
            //                        }
            //                        pcl::io::savePCDFileBinary(crop_result_path + "/orig_cluster" + std::to_string(po) + "_cluster" + std::to_string(c) +".pcd", *cloud_cluster_orig);

            //                        ColorHistogram hist;
            //                        color_corr = hist.colorCorr(static_overlap_cloud, cloud_cluster_orig);
            //                        std::cout << "Color correlation: " << color_corr << " for plane " << po << "; Cluster " << std::to_string(c) << std::endl;

            //                        if (color_corr < 0.25) { //if color is not very similar although the overlap is reasonable high --> new object
            //                            PlaneWithObjInd res_object;
            //                            res_object.plane = plane_object.plane;
            //                            res_object.obj_indices = cloud_indices;
            //                            verified_objects.push_back(res_object);
            //                            std::cout << "Plane " << po << "; Cluster " << std::to_string(c) << " gets added." << std::endl;
            //                        }
            //                    }
            //                }
            //}
        } else {
            std::cerr << "Plane ICP did not converge for cluster " << c << std::endl;
        }
    }

    return verified_objects;
}
