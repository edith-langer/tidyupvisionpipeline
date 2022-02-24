#include "local_object_verification.h"

//LocalObjectVerification::LocalObjectVerification(pcl::PointCloud<PointNormal>::ConstPtr ref_object, //pcl::PointCloud<PointNormal>::Ptr ref_plane,
//                                                 pcl::PointCloud<PointNormal>::ConstPtr curr_object, //pcl::PointCloud<PointNormal>::Ptr curr_plane,
//                                                 LocalObjectVerificationParams params)
//{
//    pcl::copyPointCloud(*ref_object, *ref_object_);
//    pcl::copyPointCloud(*curr_object, *curr_object_);
//    //this->ref_object_ = ref_object;
//    //this->curr_object_ = curr_object;
//    //    this->ref_plane_ = ref_plane;
//    //    this->curr_plane_ = curr_plane;

//    params_ = params;

//    this->debug_output_path="";
//}

void LocalObjectVerification::setDebugOutputPath (std::string debug_output_path) {
    this->debug_output_path = debug_output_path;
}

LVResult LocalObjectVerification::computeLV() {
    LVResult result;
    std::cout << "Perform Local Verification" << std::endl;


    std::cout << "Ref cluster has " << ref_object_->size() << " points." << std::endl;
    pcl::io::savePCDFileBinary(debug_output_path + "/ref_object.pcd", *ref_object_);

    std::cout << "Curr cluster has " << curr_object_->size() << " points." << std::endl;
    pcl::io::savePCDFileBinary(debug_output_path + "/curr_object.pcd", *curr_object_);

    //use that plane to align the cluster in z-direction
    //    pcl::io::savePCDFileBinary(debug_output_path + "/ref_supp_plane.pcd", *ref_plane_);
    //    pcl::io::savePCDFileBinary(debug_output_path + "/curr_supp_plane.pcd", *curr_plane_);

    //crop the supporting plane to improve ICP speed
    //get min and max values
    //        PointNormal minPt_object, maxPt_object;
    //        pcl::getMinMax3D (*curr_object_, minPt_object, maxPt_object);

    //        //crop curr and ref plane
    //        float object_plane_margin = 0.1;
    //        pcl::PassThrough<PointNormal> pass_plane_object;
    //        pass_plane_object.setInputCloud(ref_plane_);
    //        pass_plane_object.setFilterFieldName("x");
    //        pass_plane_object.setFilterLimits(minPt_object.x - object_plane_margin, maxPt_object.x + object_plane_margin);
    //        pass_plane_object.setKeepOrganized(false);
    //        pass_plane_object.filter(*ref_plane_);
    //        pass_plane_object.setInputCloud(curr_plane_);
    //        pass_plane_object.filter(*curr_plane_);
    //        pass_plane_object.setFilterFieldName("y");
    //        pass_plane_object.setFilterLimits(minPt_object.y - object_plane_margin, maxPt_object.y + object_plane_margin);
    //        pass_plane_object.filter(*curr_plane_);
    //        pass_plane_object.setInputCloud(ref_plane_);
    //        pass_plane_object.filter(*ref_plane_);


    //        //this should not happen
    //        if (supporting_plane->size() == 0) { //no plane found --> flying object --> remove it
    //            //TODO
    //            std::cout << "Cluster " << std::to_string(c) << " gets removed because no supporting plane found." << std::endl;
    //            continue;
    //        }
    //        pcl::io::savePCDFileBinary(debug_output_path + "/cropped_ref_supp_plane.pcd", *ref_plane_);
    //        pcl::io::savePCDFileBinary(debug_output_path + "/cropped_curr_supp_plane.pcd", *curr_plane_);


    //        //filter nans for ICP
    //        pcl::PointCloud<PointNormal>::Ptr static_crop_filtered(new pcl::PointCloud<PointNormal>);
    //        std::vector<int> ind;
    //        pcl::removeNaNFromPointCloud(*ref_plane_, *ref_plane_, ind);
    //        pcl::removeNaNFromPointCloud(*curr_plane_, * curr_plane_, ind);
    std::vector<int> ref_nan;
    std::vector<int> curr_nan;

    pcl::PointCloud<PointNormal>::Ptr ref_object_noNans(new pcl::PointCloud<PointNormal>);
    pcl::copyPointCloud(*ref_object_, *ref_object_noNans);
    ref_object_noNans->is_dense = false;
    pcl::PointCloud<PointNormal>::Ptr curr_object_noNans(new pcl::PointCloud<PointNormal>);
    pcl::copyPointCloud(*curr_object_, *curr_object_noNans);
    curr_object_noNans->is_dense = false;

    pcl::removeNaNFromPointCloud(*ref_object_noNans, *ref_object_noNans, ref_nan);
    pcl::removeNaNFromPointCloud(*curr_object_noNans, *curr_object_noNans, curr_nan);

    std::cout << "Ref cluster has " << ref_object_noNans->size() << " points." << std::endl;
    std::cout << "Curr cluster has " << curr_object_noNans->size() << " points." << std::endl;

    //TODO: does a two-step alignment process similar to IROS2020 makes sense?
    //the reconstructions are quite small and should be good enough for alignment

    //ICP alignment with x,y,z,z-axis alignment
    pcl::registration::WarpPointRigid4D<PointNormal, PointNormal>::Ptr warp_fcn_4d (new pcl::registration::WarpPointRigid4D<PointNormal, PointNormal>);

    // Create a TransformationEstimationLM object, and set the warp to it
    pcl::registration::TransformationEstimationLM<PointNormal, PointNormal>::Ptr te (new pcl::registration::TransformationEstimationLM<PointNormal, PointNormal>);
    te->setWarpFunction (warp_fcn_4d);


    pcl::PointCloud<PointNormal>::Ptr curr_object_registered(new pcl::PointCloud<PointNormal>());
    pcl::IterativeClosestPoint<PointNormal, PointNormal> icp;
    icp.setTransformationEstimation(te);
    icp.setInputSource(curr_object_noNans);
    icp.setInputTarget(ref_object_noNans);
    icp.setMaxCorrespondenceDistance(params_.icp_max_corr_dist_plane);
    icp.setRANSACOutlierRejectionThreshold(params_.icp_ransac_thr);
    icp.setMaximumIterations(params_.icp_max_iter);
    icp.setTransformationEpsilon (1e-9);
    //icp.setTransformationRotationEpsilon(1 - 1e-15); //epsilon is the cos(angle)
    icp.align(*curr_object_registered);

    std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;

    if (icp.hasConverged()) {
        result.transform_obj_to_model = icp.getFinalTransformation();
        v4r::apps::PPFRecognizerParameter params;
        FitnessScoreStruct fitness_score  = ObjectMatching::computeModelFitness(curr_object_registered, ref_object_noNans, params);
        result.fitness_score = fitness_score;
        float confidence = std::min(fitness_score.object_conf, fitness_score.model_conf);
        pcl::io::savePCDFileBinary(debug_output_path + "/curr_object_aligned.pcd", *curr_object_registered);
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << confidence;
        std::string conf_str = stream.str();
        pcl::io::savePCDFileBinary(debug_output_path + "/ref_object_conf_" +conf_str + ".pcd", *ref_object_noNans);

        if (confidence > params_.min_score_thr) {
            //remove points from MODEL object input cloud
            SceneDifferencingPoints scene_diff(params_.diff_dist);
            std::vector<int> diff_ind;
            std::vector<int> corresponding_ind;
            SceneDifferencingPoints::Cloud::Ptr ref_diff_cloud = scene_diff.computeDifference(ref_object_noNans, curr_object_registered, diff_ind, corresponding_ind);

            //remove very small clusters from the diff
            std::vector<int> small_cluster_ind;
            ObjectMatching::clusterOutliersBySize(ref_diff_cloud, small_cluster_ind, 0.014, min_object_size_ds);
            for (int i = small_cluster_ind.size() - 1; i >= 0; i--) { //small cluster ind is sorted ascending
                diff_ind.erase(diff_ind.begin() + small_cluster_ind[i]);
            }

            if (isObjectUnwanted(ref_diff_cloud, min_object_volume, min_object_size_ds, std::numeric_limits<int>::max(), 0.01, 0.9)) { //if less than 100 points left, we do not split the ref object
                result.model_non_matching_cloud = pcl::PointCloud<PointNormal>::Ptr(new pcl::PointCloud<PointNormal>());
            }


            //split the ref object cloud
            else {
                //the part that is matched
                pcl::PointCloud<PointNormal>::Ptr matched_ref_part(new pcl::PointCloud<PointNormal>);
                ObjectMatching::matchedPartGrowing(ref_object_noNans, matched_ref_part, ref_diff_cloud, fitness_score.model_overlapping_pts);

                if (!matched_ref_part->empty())
                    pcl::io::savePCDFileBinary(debug_output_path + "/ref_object_partial_match.pcd", *matched_ref_part);
                if (!ref_diff_cloud->empty())
                    pcl::io::savePCDFileBinary(debug_output_path + "/ref_object_diff.pcd", *ref_diff_cloud);

                result.model_matching_cloud = matched_ref_part;
                result.model_non_matching_cloud = ref_diff_cloud;
            }

            diff_ind.clear(); corresponding_ind.clear();
            SceneDifferencingPoints::Cloud::Ptr curr_diff_cloud = scene_diff.computeDifference(curr_object_registered, ref_object_noNans, diff_ind, corresponding_ind);

            //remove very small clusters from the diff
            small_cluster_ind.clear();
            ObjectMatching::clusterOutliersBySize(curr_diff_cloud, small_cluster_ind, 0.014, min_object_size_ds);

            for (int i = small_cluster_ind.size() - 1; i >= 0; i--) {
                diff_ind.erase(diff_ind.begin() + small_cluster_ind[i]);
            }
            if (isObjectUnwanted(curr_diff_cloud, min_object_volume, min_object_size_ds, std::numeric_limits<int>::max(), 0.01, 0.9)) { //if less than 100 points left, we do not split the ref object
                //result.obj_non_matching_pts = std::vector<int>{};
                result.obj_non_matching_cloud = pcl::PointCloud<PointNormal>::Ptr(new pcl::PointCloud<PointNormal>());
            }
            //split the curr object cloud
            else {
                //the part that is matched
                pcl::PointCloud<PointNormal>::Ptr matched_curr_part(new pcl::PointCloud<PointNormal>);
                ObjectMatching::matchedPartGrowing(curr_object_registered, matched_curr_part, curr_diff_cloud, fitness_score.model_overlapping_pts);

                //transform back to original scene
                pcl::transformPointCloudWithNormals(*matched_curr_part, *matched_curr_part, icp.getFinalTransformation().inverse());
                pcl::transformPointCloudWithNormals(*curr_diff_cloud, *curr_diff_cloud, icp.getFinalTransformation().inverse());

                if (!matched_curr_part->empty())
                    pcl::io::savePCDFileBinary(debug_output_path + "/curr_object_partial_match.pcd", *matched_curr_part);
                if (!curr_diff_cloud->empty())
                    pcl::io::savePCDFileBinary(debug_output_path + "/curr_object_diff.pcd", *curr_diff_cloud);

                result.obj_matching_cloud = matched_curr_part;
                result.obj_non_matching_cloud = curr_diff_cloud;
            }
            result.is_matched=true;
            return result;
        }
    }

    result.obj_matching_cloud = pcl::PointCloud<PointNormal>::Ptr(new pcl::PointCloud<PointNormal>());
    result.obj_non_matching_cloud = pcl::PointCloud<PointNormal>::Ptr(new pcl::PointCloud<PointNormal>());
    result.model_matching_cloud = pcl::PointCloud<PointNormal>::Ptr(new pcl::PointCloud<PointNormal>());
    result.model_non_matching_cloud = pcl::PointCloud<PointNormal>::Ptr(new pcl::PointCloud<PointNormal>());

    result.transform_obj_to_model = Eigen::Matrix4f::Identity();
    result.is_matched = false;

    return result;
}
