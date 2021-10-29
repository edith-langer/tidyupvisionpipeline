#include "change_detection.h"

int DetectedObject::s_id = 0;

const float diff_dist = ds_leaf_size_LV * std::sqrt(2);
const float add_crop_static = 0.10; //the amount that should be added to each cluster in the static version when doing the crop
const float icp_max_corr_dist = 0.15;
const float icp_max_corr_dist_plane = 0.05;
const int icp_max_iter = 500;
const float icp_ransac_thr = 0.015;
const float overlap = 0.7;
const float color_weight = 0.25;
const float overlap_weight = 0.75;
const float min_score_thr = 0.7;

bool comparePoint(std::pair<PointNormal, int> p1, std::pair<PointNormal, int> p2){
    if (!pcl::isFinite(p1.first))
        return false;
    else if (!pcl::isFinite(p2.first))
        return true;
    if (p1.first.x != p2.first.x)
        return p1.first.x > p2.first.x;
    else if (p1.first.y != p2.first.y)
        return  p1.first.y > p2.first.y;
    else
        return p1.first.z > p2.first.z;
}

void ChangeDetection::init(pcl::PointCloud<PointNormal>::Ptr ref_cloud, pcl::PointCloud<PointNormal>::Ptr curr_cloud,
                           const Eigen::Vector4f &ref_plane_coeffs, const Eigen::Vector4f &curr_plane_coeffs,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr ref_convex_hull_pts, pcl::PointCloud<pcl::PointXYZ>::Ptr curr_convex_hull_pts,
                           std::string ppf_model_path, std::string output_path, std::string merge_object_parts_folder, bool perform_LV_matching) {
    ref_cloud_= ref_cloud;
    curr_cloud_ = curr_cloud;
    ref_plane_coeffs_ = ref_plane_coeffs;
    curr_plane_coeffs_ = curr_plane_coeffs;
    curr_convex_hull_pts_ = curr_convex_hull_pts;
    ref_convex_hull_pts_ = ref_convex_hull_pts;
    output_path_ = output_path;
    ppf_model_path_ = ppf_model_path;
    merge_object_parts_path_ = merge_object_parts_folder;
    do_LV_before_matching_ = perform_LV_matching;
}


void ChangeDetection::compute(std::vector<DetectedObject> &ref_result, std::vector<DetectedObject> &curr_result) {

    if (ref_cloud_->empty() && curr_cloud_->empty()) {
        std::cerr << "You have to call ChangeDetection::init before calling ChangeDetection::compute !!" << std::endl;
        return;
    }

    refineNormals(curr_cloud_);
    refineNormals(ref_cloud_);



    //----------------------------downsample input clouds for faster computation-------
    pcl::PointCloud<PointNormal>::Ptr curr_cloud_downsampled (new pcl::PointCloud<PointNormal>);
    pcl::PointCloud<PointNormal>::Ptr ref_cloud_downsampled (new pcl::PointCloud<PointNormal>);
    curr_cloud_downsampled = downsampleCloudVG(curr_cloud_, ds_leaf_size_LV);
    ref_cloud_downsampled = downsampleCloudVG(ref_cloud_, ds_leaf_size_LV);

    //---------------------------------------------------------------------------------


    //call computeObjectsOnPlanes twice with different input.
    //The result contains points of clusters detected on top of the main plane
    //---Current objects------
    std::string curr_res_path =  output_path_ + "/curr_cloud/";
    boost::filesystem::create_directories(curr_res_path);
    std::vector<PlaneWithObjInd> curr_objects_from_plane = getObjectsFromPlane(curr_cloud_downsampled, curr_plane_coeffs_, curr_convex_hull_pts_,
                                                                    curr_checked_plane_point_cloud_, curr_res_path);


    //---Reference objects---
    std::string ref_res_path =  output_path_ + "/ref_cloud/";
    boost::filesystem::create_directories(ref_res_path);
    std::vector<PlaneWithObjInd> ref_objects_from_plane = getObjectsFromPlane(ref_cloud_downsampled, ref_plane_coeffs_, ref_convex_hull_pts_,
                                                                   ref_checked_plane_point_cloud_, ref_res_path);



    //hopefully remove objects from ref_objects that were detected because of reconstruction inaccuracies
    pcl::PointCloud<PointNormal>::Ptr curr_non_object_points (new pcl::PointCloud<PointNormal>);
    pcl::copyPointCloud(*curr_cloud_downsampled, *curr_non_object_points);
    for (size_t i = 0; i < curr_objects_from_plane.size(); i++) {
        pcl::ExtractIndices<PointNormal> extract;
        extract.setInputCloud (curr_non_object_points);
        pcl::PointIndices::Ptr ind(new pcl::PointIndices);
        ind->indices = curr_objects_from_plane[i].obj_indices;
        extract.setIndices (ind);
        extract.setNegative (true);
        extract.setKeepOrganized(true);
        if (i == curr_objects_from_plane.size()-1)
            extract.setKeepOrganized(false); //in the last round we can destroy the index order
        extract.filter(*curr_non_object_points);
    }
    //could be empty if the method was called with only one valid plane reconstruction
    if (!curr_non_object_points->empty()) {
        pcl::io::savePCDFileBinary(ref_res_path + "/curr_non_object_points.pcd", *curr_non_object_points);
        matchAndRemoveObjects(curr_non_object_points, ref_cloud_downsampled, ref_objects_from_plane);
    }

    //hopefully remove objects from curr_objects that were detected because of reconstruction inaccuracies
    pcl::PointCloud<PointNormal>::Ptr ref_non_object_points (new pcl::PointCloud<PointNormal>);
    pcl::copyPointCloud(*ref_cloud_downsampled, *ref_non_object_points);
    for (size_t i = 0; i < ref_objects_from_plane.size(); i++) {
        pcl::ExtractIndices<PointNormal> extract;
        extract.setInputCloud (ref_non_object_points);
        pcl::PointIndices::Ptr ind(new pcl::PointIndices);
        ind->indices = ref_objects_from_plane[i].obj_indices;
        extract.setIndices (ind);
        extract.setNegative (true);
        extract.setKeepOrganized(true);
        if (i == ref_objects_from_plane.size()-1)
            extract.setKeepOrganized(false); //in the last round we can destroy the index order
        extract.filter(*ref_non_object_points);
    }
    //could be empty if the method was called with only one valid plane reconstruction
    if (!ref_non_object_points->empty()) {
        pcl::io::savePCDFileBinary(curr_res_path + "/ref_non_object_points.pcd", *ref_non_object_points);
        matchAndRemoveObjects(ref_non_object_points, curr_cloud_downsampled, curr_objects_from_plane);
    }

    if (curr_objects_from_plane.size() != 0) {
        pcl::PointCloud<PointNormal>::Ptr curr_obj_cloud = fromObjectVecToObjectCloud(curr_objects_from_plane, curr_cloud_downsampled);
        pcl::io::savePCDFileBinary(curr_res_path + "/objects_after_matching_non_obj.pcd", *curr_obj_cloud);
    }
    if (ref_objects_from_plane.size() != 0) {
        pcl::PointCloud<PointNormal>::Ptr ref_obj_cloud = fromObjectVecToObjectCloud(ref_objects_from_plane, ref_cloud_downsampled);
        pcl::io::savePCDFileBinary(ref_res_path + "/objects_after_matching_non_obj.pcd", *ref_obj_cloud);
    }

    //region growing
    if (curr_objects_from_plane.size() > 0) {
        objectRegionGrowing(curr_cloud_downsampled, curr_objects_from_plane);  //this removes very big clusters after growing
        mergeObjects(curr_objects_from_plane); //in case an object was detected several times (disjoint sets originally, but prob. overlapping after region growing)
        pcl::PointCloud<PointNormal>::Ptr novel_objects_cloud = fromObjectVecToObjectCloud(curr_objects_from_plane, curr_cloud_downsampled);
        pcl::io::savePCDFileBinary(curr_res_path + "/result_after_objectGrowing.pcd", *novel_objects_cloud);
    }

    if (ref_objects_from_plane.size() > 0) {
        objectRegionGrowing(ref_cloud_downsampled, ref_objects_from_plane);  //this removes very big clusters after growing
        mergeObjects(ref_objects_from_plane); //in case an object was detected several times (disjoint sets originally, but prob. overlapping after region growing)
        pcl::PointCloud<PointNormal>::Ptr disappeared_objects_cloud = fromObjectVecToObjectCloud(ref_objects_from_plane, ref_cloud_downsampled);
        pcl::io::savePCDFileBinary(ref_res_path + "/result_after_objectGrowing.pcd", *disappeared_objects_cloud);
    }

    //----------------Upsample again to have the objects and planes in full resolution----------
    //    upsampleObjectsAndPlanes(curr_cloud_, curr_cloud_downsampled, curr_objects, ds_leaf_size_LV, curr_res_path);
    //    pcl::PointCloud<PointNormal>::Ptr  novel_objects_cloud = fromObjectVecToObjectCloud(curr_objects, curr_cloud_downsampled);
    //    novel_objects_cloud = fromObjectVecToObjectCloud(curr_objects, curr_cloud_);
    //    pcl::io::savePCDFileBinary(curr_res_path + "/eval_sem_plane_final_orig.pcd", *novel_objects_cloud);

    //    //----------------Upsample again to have the objects and planes in full resolution----------
    //    upsampleObjectsAndPlanes(ref_cloud_, ref_cloud_downsampled, ref_objects, ds_leaf_size_LV, ref_res_path);
    //    pcl::PointCloud<PointNormal>::Ptr disappeared_objects_cloud = fromObjectVecToObjectCloud(ref_objects, ref_cloud_downsampled);
    //    disappeared_objects_cloud = fromObjectVecToObjectCloud(ref_objects, ref_cloud_);
    //    pcl::io::savePCDFileBinary(ref_res_path + "/eval_sem_plane_final_orig.pcd", *disappeared_objects_cloud);


    //create detectedobjects
    std::vector<DetectedObject> curr_obj_vec, ref_obj_vec;
    for (size_t i = 0; i < curr_objects_from_plane.size(); i++) {
        DetectedObject obj = fromPlaneIndObjToDetectedObject(curr_cloud_downsampled, curr_objects_from_plane[i]);
        curr_obj_vec.push_back(obj);
    }
    for (size_t i = 0; i < ref_objects_from_plane.size(); i++) {
        DetectedObject obj = fromPlaneIndObjToDetectedObject(ref_cloud_downsampled, ref_objects_from_plane[i]);
        ref_obj_vec.push_back(obj);
    }


    //------------------------------LOCAL VERIFICATION IF WANTED--------------------------------------------------------


    if (do_LV_before_matching_ && curr_obj_vec.size() != 0 && ref_obj_vec.size() != 0) {
        performLV(ref_obj_vec, curr_obj_vec);

        pcl::PointCloud<PointNormal>::Ptr ref_obj_cloud = fromDetObjectVecToCloud(ref_obj_vec, false);
        pcl::PointCloud<PointNormal>::Ptr curr_obj_cloud = fromDetObjectVecToCloud(curr_obj_vec, false);
        if (!curr_obj_cloud->empty())
            pcl::io::savePCDFileBinary(curr_res_path + "/objects_after_LV.pcd", *curr_obj_cloud);
        if (!ref_obj_cloud->empty())
            pcl::io::savePCDFileBinary(ref_res_path + "/objects_after_LV.pcd", *ref_obj_cloud);
    }

    if (curr_obj_vec.size() > 0) {
        //filter out objects with similar color to supporting plane and are planar
        //std::string objects_corr_planar_path = curr_res_path + "/objects_with_corr_planar/";
        //boost::filesystem::create_directories(objects_corr_planar_path);
        //filterVerticalPlanes(curr_objects, curr_cloud_downsampled, objects_corr_planar_path);
        //pcl::PointCloud<PointNormal>::Ptr  novel_objects_cloud = fromObjectVecToObjectCloud(curr_objects, curr_cloud_downsampled);
        //pcl::io::savePCDFileBinary(curr_res_path + "/result_after_filtering_planar_objects.pcd", *novel_objects_cloud);
        //----------------Upsample again to have the objects and planes in full resolution----------
        upsampleObjectsAndPlanes(curr_cloud_, curr_obj_vec, ds_leaf_size_LV, curr_res_path);
        pcl::PointCloud<PointNormal>::Ptr novel_objects_cloud = fromDetObjectVecToCloud(curr_obj_vec, false);
        if (!novel_objects_cloud->empty())
            pcl::io::savePCDFileBinary(curr_res_path + "/upsampled_objects.pcd", *novel_objects_cloud);

        filterUnwantedObjects(curr_obj_vec, min_object_volume, min_object_size_ds, max_object_size_ds);
        novel_objects_cloud = fromDetObjectVecToCloud(curr_obj_vec, false);
        if (!novel_objects_cloud->empty())
            pcl::io::savePCDFileBinary(curr_res_path + "/final_objects.pcd", *novel_objects_cloud);


        //save each object including its supporting plane
        std::string objects_with_planes_path = curr_res_path + "/final_objects_with_planes/";
        boost::filesystem::create_directories(objects_with_planes_path);
        saveObjectsWithPlanes(objects_with_planes_path, curr_obj_vec); //save only objects with size > 15 (similar to cleanResult())
    }

    if (ref_obj_vec.size() > 0) {
        //filter out objects with similar color to supporting plane and are planar
        //std::string objects_corr_planar_path = ref_res_path + "/objects_with_corr_planar/";
        //boost::filesystem::create_directories(objects_corr_planar_path);
        //filterVerticalPlanes(ref_objects_from_plane, ref_cloud_downsampled, objects_corr_planar_path);
        //pcl::PointCloud<PointNormal>::Ptr disappeared_objects_cloud = fromObjectVecToObjectCloud(ref_objects_from_plane, ref_cloud_downsampled);
        //pcl::io::savePCDFileBinary(ref_res_path + "/result_after_filtering_planar_objects.pcd", *disappeared_objects_cloud);
        //----------------Upsample again to have the objects and planes in full resolution----------
        upsampleObjectsAndPlanes(ref_cloud_, ref_obj_vec, ds_leaf_size_LV, ref_res_path);
        pcl::PointCloud<PointNormal>::Ptr disappeared_objects_cloud = fromDetObjectVecToCloud(ref_obj_vec, false);
        if (!disappeared_objects_cloud->empty())
            pcl::io::savePCDFileBinary(ref_res_path + "/upsampled_objects.pcd", *disappeared_objects_cloud);

        filterUnwantedObjects(ref_obj_vec, min_object_volume, min_object_size_ds, max_object_size_ds);
        disappeared_objects_cloud = fromDetObjectVecToCloud(ref_obj_vec, false);
        if (!disappeared_objects_cloud->empty())
            pcl::io::savePCDFileBinary(ref_res_path + "/final_objects.pcd", *disappeared_objects_cloud);


        //save each object including its supporting plane
        std::string objects_with_planes_path = ref_res_path + "/final_objects_with_planes/";
        boost::filesystem::create_directories(objects_with_planes_path);
        saveObjectsWithPlanes(objects_with_planes_path, ref_obj_vec); //save only objects with size > 15 (similar to cleanResult())
    }



    //recompute Fitness score with upsampled version (there should be exactly the same number of static ref and curr objects)
//    v4r::apps::PPFRecognizerParameter params;
//    for (size_t i = 0; i < ref_obj_static.size(); i++) {
//        pcl::PointCloud<PointNormal>::Ptr object_aligned (new pcl::PointCloud<PointNormal>);
//        pcl::transformPointCloudWithNormals(*curr_obj_static[i].getObjectCloud(), *object_aligned, ref_obj_static[i].match_.transform);
//        FitnessScoreStruct fitness_score = ObjectMatching::computeModelFitness(object_aligned, ref_obj_static[i].getObjectCloud(), params);
//        ref_obj_static[i].match_.fitness_score = fitness_score;
//        curr_obj_static[i].match_.fitness_score = fitness_score;
//    }



    //------------------Match detected objects against each other with PPF-----------------------
    //depending on if LV was performed before, static objects can be matched
    std::string debug_model_path = output_path_ + "/model_objects/";
    for (DetectedObject obj : ref_obj_vec) {
        pcl::PointCloud<PointNormal>::Ptr refined_normals_cloud (new pcl::PointCloud<PointNormal>);
        pcl::copyPointCloud(*(obj.getObjectCloud()), *refined_normals_cloud);
        refineNormals(refined_normals_cloud);
        obj.setObjectCloud(refined_normals_cloud);

        std::string debug_obj_folder = debug_model_path + std::to_string(obj.getID()); //PPF uses the folder name as model_id!
        boost::filesystem::create_directories(debug_obj_folder);
        std::string obj_folder = ppf_model_path_ + std::to_string(obj.getID()); //PPF uses the folder name as model_id!
        boost::filesystem::create_directories(obj_folder);
        pcl::io::savePCDFile(debug_obj_folder + "/3D_model.pcd", *(obj.getObjectCloud())); //each detected reference object is saved in a folder for further use with PPF
        pcl::io::savePCDFileBinary(debug_obj_folder + "/plane.pcd", *(obj.plane_cloud_));
        pcl::io::savePCDFile(obj_folder + "/3D_model.pcd", *(obj.getObjectCloud()));
        obj.object_folder_path_ = obj_folder;
    }

    filterUnwantedObjects(ref_obj_vec, min_object_volume, min_object_size_ds); //0.05^3

    if (curr_obj_vec.size() == 0) { //all objects removed from ref scene
        for (size_t i = 0; i < ref_obj_vec.size(); i++) {
            ref_obj_vec[i].state_ = ObjectState::REMOVED;
        }
        ref_result = ref_obj_vec;
        mergeObjectParts(ref_result, merge_object_parts_path_);
        return;
    }

    for (DetectedObject obj : curr_obj_vec) {
        pcl::PointCloud<PointNormal>::Ptr refined_normals_cloud (new pcl::PointCloud<PointNormal>);
        pcl::copyPointCloud(*(obj.getObjectCloud()), *refined_normals_cloud);
        refineNormals(refined_normals_cloud);
        obj.setObjectCloud(refined_normals_cloud);

        std::string debug_obj_folder = output_path_ + "/current_det_objects/" + std::to_string(obj.getID());
        boost::filesystem::create_directories(debug_obj_folder);
        pcl::io::savePCDFileBinary(debug_obj_folder + "/object.pcd", *(obj.getObjectCloud()));
        pcl::io::savePCDFileBinary(debug_obj_folder + "/plane.pcd", *(obj.plane_cloud_));
    }

    filterUnwantedObjects(curr_obj_vec, min_object_volume, min_object_size_ds);

    if (ref_obj_vec.size() == 0) { //all objects are new in curr
        for (size_t i = 0; i < curr_obj_vec.size(); i++) {
            curr_obj_vec[i].state_ = ObjectState::NEW;
        }
        curr_result = curr_obj_vec;
        mergeObjectParts(curr_result, merge_object_parts_path_);
        return;
    }

    //TODO isn't ref_result and curr_result empty at this point?
    mergeObjectParts(ref_result, merge_object_parts_path_);
    mergeObjectParts(curr_result, merge_object_parts_path_);

    filterUnwantedObjects(ref_result, min_object_volume, min_object_size_ds);
    filterUnwantedObjects(curr_result, min_object_volume, min_object_size_ds);

    //matches between same plane different timestamps
    ObjectMatching object_matching(ref_obj_vec, curr_obj_vec, ppf_model_path_, ppf_config_path_);
    std::vector<Match> matches = object_matching.compute(ref_result, curr_result);

    //region growing of static/displaced objects (should create more precise results if e.g. the model was smaller than die object or not precisely aligned
    mergeObjectParts(ref_result, merge_object_parts_path_);
    mergeObjectParts(curr_result, merge_object_parts_path_);
    filterUnwantedObjects(ref_result, min_object_volume, min_object_size_ds);
    filterUnwantedObjects(curr_result, min_object_volume, min_object_size_ds);
}

void ChangeDetection::upsampleObjectsAndPlanes(pcl::PointCloud<PointNormal>::Ptr orig_cloud, std::vector<DetectedObject> &objects, double leaf_size, std::string res_path) {
    pcl::octree::OctreePointCloudSearch<PointNormal>::Ptr octree;
    octree.reset(new pcl::octree::OctreePointCloudSearch<PointNormal>(leaf_size));
    octree->setInputCloud(orig_cloud);
    octree->addPointsFromInputCloud();

    for (size_t i = 0; i < objects.size(); i++) {
        DetectedObject &obj = objects[i];
        //upsample plane indices
        std::tuple<pcl::PointCloud<PointNormal>::Ptr, std::vector<int> > obj_cloud_ind_tuple = upsampleObjects(octree, orig_cloud, obj.plane_cloud_, res_path, i);
        obj.plane_cloud_ = std::get<0>(obj_cloud_ind_tuple);

        PointNormal minPt_object, maxPt_object;
        pcl::getMinMax3D (*(obj.plane_cloud_), minPt_object, maxPt_object);

        /// crop cloud
        float crop_margin = 0.5; //this margin is not so important it is just reducing the number of points for the next step
        pcl::PointCloud<PointNormal>::Ptr cloud_crop(new pcl::PointCloud<PointNormal>);
        pcl::PassThrough<PointNormal> pass;
        pass.setInputCloud(orig_cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(minPt_object.z, maxPt_object.z + crop_margin);
        pass.setKeepOrganized(true);
        pass.filter(*cloud_crop);

        //make the plane a little thicker. these plane inliers are then checked against object indices to remove overlapping points
        pcl::SampleConsensusModelPlane<PointNormal>::Ptr dit (new pcl::SampleConsensusModelPlane<PointNormal> (cloud_crop));
        std::vector<int> inliers;
        Eigen::Vector4f vec_coeff (obj.plane_coeffs_->values.data ()); //from pcl::ModelCoefficients to Eigen::Vector
        dit -> selectWithinDistance (vec_coeff, leaf_size/2, inliers);

        //also object indices need to be upsampled (this could lead to a state where a point is assigned to an object AND the plane!)
        obj_cloud_ind_tuple= upsampleObjects(octree, orig_cloud, obj.getObjectCloud(), res_path, i);
        std::vector<int> ind =get<1>(obj_cloud_ind_tuple);

        //remove points that are detected as planes and object from the object. we don't want plane points in the object for better feature detection
        std::unordered_set<int> ind_set(ind.begin(), ind.end());
        for (auto a : inliers) {
            ind_set.erase(a); //erase element if it exists in the set
        }

        pcl::PointIndices::Ptr pt_ind(new pcl::PointIndices);
        std::copy(ind_set.begin(), ind_set.end(), back_inserter(pt_ind->indices));

        pcl::PointCloud<PointNormal>::Ptr obj_cloud_cleaned(new pcl::PointCloud<PointNormal>);
        pcl::ExtractIndices<PointNormal> extract_object_ind;
        extract_object_ind.setInputCloud (orig_cloud);
        extract_object_ind.setIndices(pt_ind);
        extract_object_ind.setKeepOrganized(false);
        extract_object_ind.setNegative (false);
        extract_object_ind.filter (*obj_cloud_cleaned);

        obj.setObjectCloud(obj_cloud_cleaned);
    }
}

void ChangeDetection::upsampleObjectsAndPlanes(pcl::PointCloud<PointNormal>::Ptr orig_cloud, pcl::PointCloud<PointNormal>::Ptr ds_cloud,
                                               std::vector<PlaneWithObjInd> &objects, double leaf_size, std::string res_path) {
    pcl::octree::OctreePointCloudSearch<PointNormal>::Ptr octree;
    octree.reset(new pcl::octree::OctreePointCloudSearch<PointNormal>(leaf_size));
    octree->setInputCloud(orig_cloud);
    octree->addPointsFromInputCloud();
    std::cout << "Created octree" << std::endl;

    for (size_t i = 0; i < objects.size(); i++) {
        //upsample plane indices
        pcl::PointIndices::Ptr plane_indices = objects[i].plane.plane_ind;
        pcl::PointCloud<PointNormal>::Ptr plane_ds_cloud(new pcl::PointCloud<PointNormal>);
        //extract object indices from downsampled cloud
        pcl::ExtractIndices<PointNormal> extract_object_ind;
        extract_object_ind.setInputCloud (ds_cloud);
        extract_object_ind.setIndices(plane_indices);
        extract_object_ind.setKeepOrganized(false);
        extract_object_ind.setNegative (false);
        extract_object_ind.filter (*plane_ds_cloud);
        std::tuple<pcl::PointCloud<PointNormal>::Ptr, std::vector<int> > obj_cloud_ind_tuple = upsampleObjects(octree, orig_cloud, plane_ds_cloud, res_path, i);
        objects[i].plane.plane_ind->indices = std::get<1>(obj_cloud_ind_tuple);

        PointNormal minPt_object, maxPt_object;
        pcl::getMinMax3D (*std::get<0>(obj_cloud_ind_tuple), minPt_object, maxPt_object);

        /// crop cloud
        float crop_margin = 0.5; //this margin is not so important it is just reducing the number of points for the next step
        pcl::PointCloud<PointNormal>::Ptr cloud_crop(new pcl::PointCloud<PointNormal>);
        pcl::PassThrough<PointNormal> pass;
        pass.setInputCloud(orig_cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(minPt_object.z, maxPt_object.z + crop_margin);
        pass.setKeepOrganized(true);
        pass.filter(*cloud_crop);

        //make the plane a little thicker. these plane inliers are then checked against object indices to remove overlapping points
        pcl::SampleConsensusModelPlane<PointNormal>::Ptr dit (new pcl::SampleConsensusModelPlane<PointNormal> (cloud_crop));
        std::vector<int> inliers;
        Eigen::Vector4f vec_coeff (objects[i].plane.coeffs->values.data ()); //from pcl::ModelCoefficients to Eigen::Vector
        dit -> selectWithinDistance (vec_coeff, leaf_size/2, inliers);

        //also object indices need to be upsampled (this could lead to a state where a point is assigned to an object AND the plane!)
        std::vector<int> obj_indices = objects[i].obj_indices;
        pcl::PointCloud<PointNormal>::Ptr object_ds_cloud(new pcl::PointCloud<PointNormal>);
        //extract object indices from downsampled cloud
        extract_object_ind.setInputCloud (ds_cloud);
        extract_object_ind.setIndices(boost::make_shared<std::vector<int> > (obj_indices));
        extract_object_ind.setKeepOrganized(false);
        extract_object_ind.setNegative (false);
        extract_object_ind.filter (*object_ds_cloud);
        //pcl::io::savePCDFileBinary(res_path + "/ds_objects" + std::to_string(i)+ ".pcd", *object_ds_cloud);
        obj_cloud_ind_tuple= upsampleObjects(octree, orig_cloud, object_ds_cloud, res_path, i);
        std::vector<int> ind =get<1>(obj_cloud_ind_tuple);

        //remove points that are detected as planes and object from the object. we don't want plane points in the object for better feature detection
        std::unordered_set<int> set(ind.begin(), ind.end());
        for (auto a : inliers) {
            set.erase(a); //erase element if it exists in the set
        }
        objects[i].obj_indices.clear();
        std::copy(set.begin(), set.end(), back_inserter(objects[i].obj_indices));

    }
}

std::tuple<pcl::PointCloud<PointNormal>::Ptr, std::vector<int> > ChangeDetection::upsampleObjects(pcl::octree::OctreePointCloudSearch<PointNormal>::Ptr octree, pcl::PointCloud<PointNormal>::ConstPtr orig_input_cloud,
                                                                                                  pcl::PointCloud<PointNormal>::ConstPtr objects_ds_cloud, std::string output_path, int counter) {

    /// instead we just add all points wihtin 1 cm to the object points
    std::vector<int> nn_indices;
    std::vector<float> nn_distances;
    std::vector<int> orig_object_ind;

    for (size_t i = 0; i < objects_ds_cloud->points.size(); ++i) {
        if (!pcl::isFinite(objects_ds_cloud->points[i]))
            continue;

        const PointNormal &p_object = objects_ds_cloud->points[i];

        if (octree->radiusSearch(p_object, octree->getResolution(), nn_indices, nn_distances) > 0){
            for (size_t j = 0; j < nn_indices.size(); j++) {
                orig_object_ind.push_back(nn_indices[j]);
            }
        }
    }
    std::sort(orig_object_ind.begin(), orig_object_ind.end());
    orig_object_ind.erase(std::unique(orig_object_ind.begin(), orig_object_ind.end()), orig_object_ind.end());

    pcl::PointCloud<PointNormal>::Ptr orig_object(new pcl::PointCloud<PointNormal>);
    //extract object indices from original cloud
    pcl::ExtractIndices<PointNormal> extract_object_ind;
    extract_object_ind.setInputCloud (orig_input_cloud);
    extract_object_ind.setIndices(boost::make_shared<std::vector<int> > (orig_object_ind));
    extract_object_ind.setKeepOrganized(true);
    extract_object_ind.setNegative (false);
    extract_object_ind.filter (*orig_object);
    pcl::io::savePCDFileBinary(output_path + "/object_orig" + std::to_string(counter)+ ".pcd", *orig_object);

    std::tuple<pcl::PointCloud<PointNormal>::Ptr, std::vector<int> > obj_cloud_ind_tuple;
    obj_cloud_ind_tuple = std::make_tuple(orig_object, orig_object_ind);

    return obj_cloud_ind_tuple;
}

//upsample objects and region growing; filter big objects
void ChangeDetection::objectRegionGrowing(pcl::PointCloud<PointNormal>::Ptr cloud, std::vector<PlaneWithObjInd> &objects, int max_object_size) {
    for (size_t i = 0; i < objects.size(); i++) {
        pcl::PointCloud<PointNormal>::Ptr object_cloud(new pcl::PointCloud<PointNormal>);
        for (size_t p = 0; p < objects[i].obj_indices.size(); p++) {
            object_cloud->points.push_back(cloud->points[objects[i].obj_indices[p]]);
        }
        PointNormal minPt_object, maxPt_object;
        pcl::getMinMax3D (*object_cloud, minPt_object, maxPt_object);

        /// crop cloud
        float crop_margin = 0.5; //choosing a larger crop margin lets objects grow larger (especially FP objects) that are then filtered out
        pcl::PointCloud<PointNormal>::Ptr cloud_crop(new pcl::PointCloud<PointNormal>);
        pcl::PassThrough<PointNormal> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(minPt_object.x - crop_margin, maxPt_object.x + crop_margin);
        pass.setKeepOrganized(true);
        pass.filter(*cloud_crop);
        pass.setInputCloud(cloud_crop);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(minPt_object.y - crop_margin, maxPt_object.y + crop_margin);
        pass.setKeepOrganized(true);
        pass.filter(*cloud_crop);
        pass.setInputCloud(cloud_crop);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(minPt_object.z, maxPt_object.z + crop_margin);
        pass.setKeepOrganized(true);
        pass.filter(*cloud_crop);

        //remove supporting plane
        pcl::ExtractIndices<PointNormal> extract_plane_ind;
        extract_plane_ind.setInputCloud (cloud_crop);
        extract_plane_ind.setIndices(objects[i].plane.plane_ind);
        extract_plane_ind.setKeepOrganized(true);
        extract_plane_ind.setNegative (true);
        extract_plane_ind.filter (*cloud_crop);
        //we do not use the plane points directly because there is a chance that they do not cover the whole plane.
        //This is because the plane extraction part operates on semantic segmentation and if one plane is assigned to several labels, they are not part of the detected plane.
        //        pcl::SampleConsensusModelPlane<PointNormal>::Ptr dit (new pcl::SampleConsensusModelPlane<PointNormal> (cloud_crop));
        //        std::vector<int> inliers;
        //        Eigen::Vector4f vec_coeff (objects[i].plane.coeffs->values.data ()); //from pcl::ModelCoefficients to Eigen::Vector
        //        dit -> selectWithinDistance (vec_coeff, 0.01, inliers);
        //        PointNormal nan_point;
        //        nan_point.x = nan_point.y = nan_point.z = std::numeric_limits<float>::quiet_NaN();
        //        for (size_t p = 0; p < inliers.size(); p++) {
        //            cloud_crop->points[inliers[p]] = nan_point;
        //        }

        //pcl::io::savePCDFileBinary("/home/edith/crop_no_plane" + std::to_string(i) + ".pcd", *cloud_crop);
        //pcl::io::savePCDFileBinary("/home/edith/object_cloud" + std::to_string(i) + ".pcd", *object_cloud);


        pcl::PointCloud<pcl::Normal>::Ptr scene_normals (new pcl::PointCloud<pcl::Normal>);
        pcl::copyPointCloud(*cloud_crop, *scene_normals);

        //call the region growing method and extract upsampled object
        RegionGrowing<PointNormal, PointNormal> region_growing(cloud_crop, object_cloud, scene_normals, true);
        std::vector<int> orig_object_ind = region_growing.compute();

        pcl::PointCloud<PointNormal>::Ptr orig_object_cloud(new pcl::PointCloud<PointNormal>);
        //extract object indices from downsampled cloud
        pcl::ExtractIndices<PointNormal> extract_object_ind;
        extract_object_ind.setInputCloud (cloud_crop);
        extract_object_ind.setIndices(boost::make_shared<std::vector<int> > (orig_object_ind));
        extract_object_ind.setKeepOrganized(false);
        extract_object_ind.setNegative (false);
        extract_object_ind.filter (*orig_object_cloud);

        PointNormal minPt_orig_object, maxPt_orig_object;
        pcl::getMinMax3D (*orig_object_cloud, minPt_orig_object, maxPt_orig_object);

        if (std::abs(maxPt_orig_object.x - minPt_orig_object.x) < 10*std::abs(maxPt_object.x - minPt_object.x) &&
                std::abs(maxPt_orig_object.y - minPt_orig_object.y) < 10*std::abs(maxPt_object.y - minPt_object.y) &&
                std::abs(maxPt_orig_object.z - minPt_orig_object.z) < 10*std::abs(maxPt_object.z - minPt_object.z) ) //otherwise something went wrong with the region growing (expanded too much)
            objects[i].obj_indices = orig_object_ind;
    }
//    std::cout << "Original number of objects: " << objects.size() << ", objects after region growing (big ones got filtered): ";
//    objects.erase(
//                std::remove_if(
//                    objects.begin(),
//                    objects.end(),
//                    [&](PlaneWithObjInd const & p) { return p.obj_indices.size() > max_object_size; }
//                ),
//            objects.end()
//            );
//    std::cout << objects.size() << std::endl;
}

pcl::PointCloud<PointNormal>::Ptr ChangeDetection::fromObjectVecToObjectCloud(const std::vector<PlaneWithObjInd> objects, pcl::PointCloud<PointNormal>::Ptr cloud, bool keepOrganized) {
    PointNormal nan_point;
    nan_point.x = nan_point.y = nan_point.z = std::numeric_limits<float>::quiet_NaN();

    pcl::PointCloud<PointNormal>::Ptr objects_cloud (new pcl::PointCloud<PointNormal>(cloud->width, cloud->height, nan_point));
    objects_cloud->is_dense=false;
    for (size_t o = 0; o < objects.size(); o++) {
        std::vector<int> obj_ind =  objects[o].obj_indices;
        for (size_t i = 0; i < obj_ind.size(); i++) {
            PointNormal &p = cloud->points.at(obj_ind.at(i));
            objects_cloud->points[obj_ind[i]] = p;
        }
    }

    if (!keepOrganized) {
        std::vector<int> nan_vec;
        pcl::removeNaNFromPointCloud(*objects_cloud, *objects_cloud, nan_vec);
        objects_cloud->is_dense=true;
    }
    return objects_cloud;
}

void ChangeDetection::saveObjectsWithPlanes(std::string path, const std::vector<DetectedObject> objects) {
    PointLabel nan_point;
    nan_point.x = nan_point.y = nan_point.z = std::numeric_limits<float>::quiet_NaN();

    for (size_t o = 0; o < objects.size(); o++) {
        pcl::PointCloud<PointLabel>::Ptr res_cloud (new pcl::PointCloud<PointLabel>);
        const DetectedObject &obj = objects[o];
        for (size_t i = 0; i < obj.getObjectCloud()->points.size(); i++) {
            const PointNormal &p = obj.getObjectCloud()->points.at(i);
            PointLabel p_l;
            p_l.x = p.x; p_l.y = p.y; p_l.z = p.z;
            p_l.rgb = p.rgb;
            p_l.label = 0;  //label 0 for object
            res_cloud->points.push_back(p_l);
        }
        for (size_t i = 0; i < obj.plane_cloud_->points.size(); i++) {
            const PointNormal &p = obj.plane_cloud_->points.at(i);
            PointLabel p_l;
            p_l.x = p.x; p_l.y = p.y; p_l.z = p.z;
            p_l.rgb = p.rgb;
            p_l.label = 1;  //label 1 for plane
            res_cloud->points.push_back(p_l);
        }
        pcl::io::savePCDFileBinary(path + "object" + std::to_string(o) + ".pcd", *res_cloud);
    }
}

void ChangeDetection::saveObjectsWithPlanes(std::string path, const std::vector<PlaneWithObjInd> objects, pcl::PointCloud<PointNormal>::Ptr cloud) {
    PointLabel nan_point;
    nan_point.x = nan_point.y = nan_point.z = std::numeric_limits<float>::quiet_NaN();

    for (size_t o = 0; o < objects.size(); o++) {
        pcl::PointCloud<PointLabel>::Ptr res_cloud (new pcl::PointCloud<PointLabel>(cloud->width, cloud->height, nan_point));
        std::vector<int> obj_ind =  objects[o].obj_indices;
        if (obj_ind.size() < 15)
            continue;
        for (size_t i = 0; i < obj_ind.size(); i++) {
            const PointNormal &p = cloud->points.at(obj_ind.at(i));
            res_cloud->points[obj_ind[i]].x = p.x;
            res_cloud->points[obj_ind[i]].y = p.y;
            res_cloud->points[obj_ind[i]].z = p.z;
            res_cloud->points[obj_ind[i]].rgb = p.rgb;
            res_cloud->points[obj_ind[i]].label = 0; //label 0 for object
        }
        const std::vector<int> &plane_ind = objects[o].plane.plane_ind->indices;
        for (size_t i = 0; i < plane_ind.size(); i++) {
            const PointNormal &p = cloud->points.at(plane_ind.at(i));
            res_cloud->points[plane_ind[i]].x = p.x;
            res_cloud->points[plane_ind[i]].y = p.y;
            res_cloud->points[plane_ind[i]].z = p.z;
            res_cloud->points[plane_ind[i]].rgb = p.rgb;
            res_cloud->points[plane_ind[i]].label = 1; //label 1 for plane
        }
        pcl::io::savePCDFileBinary(path + "object" + std::to_string(o) + ".pcd", *res_cloud);
    }
}

//this method changes the object vector!
void ChangeDetection::mergeObjects(std::vector<PlaneWithObjInd>& objects) {
    std::cout << "Number of objects before merging " << objects.size() << std::endl;
    //sort point indices for further operations
    for (size_t o = 0; o < objects.size(); o++) {
        std::sort(objects[o].obj_indices.begin(), objects[o].obj_indices.end());
    }

    //iterate through all object and check if any other object has a high enough overlap of same point indices
    //if so, merge the point indices of the two objects (without duplicates) and remove the later one.
    for(std::vector<PlaneWithObjInd>::iterator it = objects.begin(), end = objects.end(); it != end;) {
        std::vector<PlaneWithObjInd>::iterator it_inner = it;
        for(++it_inner; it_inner != end;) {
            std::vector<int> &o_ind = it->obj_indices;
            std::vector<int> &i_ind = it_inner->obj_indices;
            std::vector<int> intersection;
            std::set_intersection(o_ind.begin(),o_ind.end(),
                                  i_ind.begin(),i_ind.end(),
                                  back_inserter(intersection));
            if (intersection.size() > 0) {
                if (o_ind.size()/intersection.size() > 0.8 || i_ind.size()/intersection.size() > 0.8) { //probably the same object
                    o_ind.insert(o_ind.end(), i_ind.begin(), i_ind.end());
                    std::sort(o_ind.begin(), o_ind.end());
                    o_ind.erase(unique(o_ind.begin(), o_ind.end()), o_ind.end());
                    it_inner = objects.erase(it_inner);
                    end = objects.end();
                }
            } else {
                it_inner++;
            }
        }
        it++;
    }
    std::cout << "Number of objects after merging " << objects.size() << std::endl;
}


int ChangeDetection::checkVerticalPlanarity(PlaneWithObjInd& object, pcl::PointCloud<PointNormal>::Ptr cloud, float plane_dist_thr) {

    pcl::PointCloud<PointNormal>::Ptr object_cloud (new pcl::PointCloud<PointNormal>);
    std::vector<int> obj_ind =  object.obj_indices;
    for (size_t i = 0; i < obj_ind.size(); i++) {
        PointNormal &p = cloud->points.at(obj_ind.at(i));
        if (pcl::isFinite(p))
            object_cloud->push_back(p);
    }
    pcl::SACSegmentation<PointNormal> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PARALLEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (plane_dist_thr);
    seg.setAxis(Eigen::Vector3f(0,0,1));
    seg.setEpsAngle(5.0f * (M_PI/180.0f));
    seg.setInputCloud (object_cloud);
    seg.segment (*inliers, *coefficients);

    std::cout << "plane points: " << inliers->indices.size() << "; object points " <<  obj_ind.size() << std::endl;

    if (inliers->indices.size() == 0)
        return 0;
    //I don't know why this is happening, but it is apprently not a correct solution and would include all points as plane inliers
    if (coefficients->values[0] == 0 && coefficients->values[1] == 0 && coefficients->values[2] == 0 && coefficients->values[3] == 0)
        return 0;

    return inliers->indices.size();
}




double ChangeDetection::checkColorSimilarityHistogram(PlaneWithObjInd& object, pcl::PointCloud<PointNormal>::Ptr cloud, std::string path, int _nr_bins) {
    int nr_bins = _nr_bins;
    pcl::PointCloud<PointNormal>::Ptr object_cloud (new pcl::PointCloud<PointNormal>);

    pcl::PointCloud<PointNormal>::Ptr cloud_wo_obj(new pcl::PointCloud<PointNormal>);
    pcl::ExtractIndices<PointNormal> extract;
    extract.setInputCloud (cloud);
    pcl::PointIndices::Ptr c_ind (new pcl::PointIndices);
    c_ind->indices =object.obj_indices;
    extract.setIndices (c_ind);
    extract.setKeepOrganized(false);
    extract.setNegative (true);
    extract.filter (*cloud_wo_obj);
    extract.setNegative (false);
    extract.filter (*object_cloud);

    //crop supp. plane to get only points around the object
    PointNormal minPt, maxPt;
    pcl::getMinMax3D(*object_cloud, minPt, maxPt);

    /// crop cloud
    float crop_margin = 0.05;
    pcl::PointCloud<PointNormal>::Ptr cloud_crop(new pcl::PointCloud<PointNormal>);
    pcl::PassThrough<PointNormal> pass;

    pass.setInputCloud(cloud_wo_obj);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(minPt.x - crop_margin, maxPt.x + crop_margin);
    pass.setKeepOrganized(false);
    pass.filter(*cloud_crop);
    pass.setInputCloud(cloud_crop);
    pass.setFilterFieldName("y");
    pass.setFilterLimits(minPt.y - crop_margin, maxPt.y + crop_margin);
    pass.filter(*cloud_crop);
    pass.setInputCloud(cloud_crop);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(minPt.z - crop_margin, maxPt.z + crop_margin);
    pass.filter(*cloud_crop);

    if (path!="") {
        pcl::io::savePCDFileBinary(path + "/object.pcd", *object_cloud);
        pcl::io::savePCDFileBinary(path + "/cloud_crop.pcd", *cloud_crop);
    }

    ColorHistogram color_hist;
    double corr = color_hist.colorCorr(object_cloud, cloud_crop, nr_bins);

    return corr;
}

std::vector<double> ChangeDetection::filterBasedOnColor(std::vector<PlaneWithObjInd>& objects, pcl::PointCloud<PointNormal>::Ptr cloud, int _nr_bins) {
    std::vector<int> obj_ind_to_del;
    std::vector<double> correlations;

    std::cout << "Number of objects before deleting based on colors " << objects.size() << std::endl;
    for (size_t o = 0; o < objects.size(); o++) {
        std::cout << "Check object number " << o << " for color similarity" << std::endl;
        correlations.push_back(checkColorSimilarityHistogram(objects[o], cloud, "", _nr_bins));

        //        pcl::PointCloud<PointNormal>::Ptr combined_cloud(new pcl::PointCloud<PointNormal>);
        //        *combined_cloud += *object_cloud;
        //        *combined_cloud += *plane_crop;
        //        pcl::io::savePCDFileBinary(path + "/object" + std::to_string(o) + "_corr_" + std::to_string(corr)+ ".pcd", *combined_cloud);
        //        pcl::io::savePCDFileBinary(path + "/object" + std::to_string(o) + "_object.pcd", *object_cloud);
        //        pcl::io::savePCDFileBinary(path + "/object" + std::to_string(o) + "plane.pcd", *plane_crop);
    }
    //now remove objects by index starting with the highest index in the list
    for (auto i = obj_ind_to_del.rbegin(); i != obj_ind_to_del.rend(); ++ i)
    {
        objects.erase(objects.begin() + *i);
    }
    std::cout << "Number of objects after deleting based on colors " << objects.size() << std::endl;
    return correlations;
}

//returns the number of plane points (plane inlier index does not correspond to points in original cloud!)
int ChangeDetection::checkPlanarity (PlaneWithObjInd& object, pcl::PointCloud<PointNormal>::Ptr cloud, float _plane_dist_thr) {
    float plane_dist_thr = _plane_dist_thr;

    pcl::PointCloud<PointNormal>::Ptr object_cloud (new pcl::PointCloud<PointNormal>);
    std::vector<int> obj_ind =  object.obj_indices;
    for (size_t i = 0; i < obj_ind.size(); i++) {
        PointNormal &p = cloud->points.at(obj_ind.at(i));
        object_cloud->push_back(p);
    }
    pcl::SACSegmentation<PointNormal> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);

    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (plane_dist_thr);
    seg.setInputCloud (object_cloud);
    seg.segment (*inliers, *coefficients);

    std::cout << "plane points: " << inliers->indices.size() << "; object points threshold " <<  0.9 * obj_ind.size() << std::endl;

    return inliers->indices.size();
}

std::vector<int> ChangeDetection::removePlanarObjects (std::vector<PlaneWithObjInd>& objects, pcl::PointCloud<PointNormal>::Ptr cloud, float _plane_dist_thr) {
    float plane_dist_thr = _plane_dist_thr;
    std::vector<int> obj_ind_to_del;
    std::vector<int> inlier_size_res;

    std::cout << "Number of objects before deleting planar objects " << objects.size() << std::endl;
    for (size_t o=0; o<objects.size(); o++)
    {
        inlier_size_res.push_back(checkPlanarity(objects[o], cloud, plane_dist_thr));
    }
    for (auto i = obj_ind_to_del.rbegin(); i != obj_ind_to_del.rend(); ++ i)
    {
        objects.erase(objects.begin() + *i);
    }
    std::cout << "Number of objects after deleting planar objects " << objects.size() << std::endl;
    return inlier_size_res;
}

void ChangeDetection::filterVerticalPlanes(std::vector<PlaneWithObjInd>& objects, pcl::PointCloud<PointNormal>::Ptr cloud, std::string path, float _plane_dist_thr) {
    std::vector<int> obj_ind_to_del;

    std::cout << "Number of objects before filterting based on planarity " << objects.size() << std::endl;
    for (size_t o = 0; o < objects.size(); o++) {
        std::cout << "Check object number " << o << std::endl;
        int plane_inliers = checkVerticalPlanarity(objects[o], cloud, _plane_dist_thr);
        double planarity = (double)plane_inliers/objects[o].obj_indices.size();
        std::cout << "Planarity: " << planarity << std::endl;
        if (planarity > 0.85 ) {
            //if ( (corr > 0.4) && (planarity >0.4) || (corr > 0.9) || (planarity > 0.9) ) {
            obj_ind_to_del.push_back(o);
            std::cout << "Deleted" << std::endl;

            pcl::PointCloud<PointNormal>::Ptr filtered_object_cloud(new pcl::PointCloud<PointNormal>);
            pcl::ExtractIndices<PointNormal> extract;
            extract.setInputCloud (cloud);
            pcl::PointIndices::Ptr c_ind (new pcl::PointIndices);
            c_ind->indices =objects[o].obj_indices;
            extract.setIndices (c_ind);
            extract.setKeepOrganized(false);
            extract.setNegative (false);
            extract.filter (*filtered_object_cloud);

            pcl::io::savePCDFileBinary(path + "/object" + std::to_string(o) + "planarity" + std::to_string(planarity) + ".pcd", *filtered_object_cloud);
        }
    }
    for (auto i = obj_ind_to_del.rbegin(); i != obj_ind_to_del.rend(); ++ i)
    {
        objects.erase(objects.begin() + *i);
    }
    std::cout << "Number of objects after filtering based on planarity " << objects.size() << std::endl;
}

void ChangeDetection::filterPlanarAndColor(std::vector<PlaneWithObjInd>& objects, pcl::PointCloud<PointNormal>::Ptr cloud, std::string path, float _plane_dist_thr, int _nr_bins) {
    std::vector<int> obj_ind_to_del;

    std::cout << "Number of objects before filterting based on planarity and color " << objects.size() << std::endl;
    for (size_t o = 0; o < objects.size(); o++) {
        std::cout << "Check object number " << o << std::endl;
        int plane_inliers = checkPlanarity(objects[o], cloud, _plane_dist_thr);
        double corr = checkColorSimilarityHistogram(objects[o], cloud, path, _nr_bins);
        double planarity = (double)plane_inliers/objects[o].obj_indices.size();
        std::cout << "Correlation: " << corr << "; Planarity: " << planarity << "; Combined threshold: " << (0.25*corr + 0.75 * planarity) << std::endl;
        if ((0.25*corr + 0.75 * planarity) > 0.9 ) {
            //if ( (corr > 0.4) && (planarity >0.4) || (corr > 0.9) || (planarity > 0.9) ) {
            obj_ind_to_del.push_back(o);
            std::cout << "Deleted" << std::endl;

            pcl::PointCloud<PointNormal>::Ptr filtered_object_cloud(new pcl::PointCloud<PointNormal>);
            pcl::ExtractIndices<PointNormal> extract;
            extract.setInputCloud (cloud);
            pcl::PointIndices::Ptr c_ind (new pcl::PointIndices);
            c_ind->indices =objects[o].obj_indices;
            extract.setIndices (c_ind);
            extract.setKeepOrganized(false);
            extract.setNegative (false);
            extract.filter (*filtered_object_cloud);

            pcl::io::savePCDFileBinary(path + "/object" + std::to_string(o) + ".pcd", *filtered_object_cloud);
        }
    }
    for (auto i = obj_ind_to_del.rbegin(); i != obj_ind_to_del.rend(); ++ i)
    {
        objects.erase(objects.begin() + *i);
    }
    std::cout << "Number of objects after filtering based on planarity and color " << objects.size() << std::endl;
}


//combine the orientation from the reconstruction with the more accurate normals computed with the points
void ChangeDetection::refineNormals(pcl::PointCloud<PointNormal>::Ptr object_cloud) {
    //compute normals
    pcl::PointCloud<pcl::Normal>::Ptr object_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<PointNormal, pcl::Normal>::Ptr ne(new pcl::NormalEstimation<PointNormal, pcl::Normal>);
    ne->setRadiusSearch(0.02);
    ne->setInputCloud(object_cloud);
    ne->compute(*object_normals);

    //check orientation
    for (size_t p = 0; p < object_cloud->size(); p++) {
        const auto normal_pre = object_cloud->at(p).getNormalVector4fMap();
        const auto normal_comp = object_normals->at(p).getNormalVector4fMap();
        float normals_dotp = normal_comp.dot(normal_pre);
        if (normals_dotp < 0) { //negative dot product = angle between the two normals is between 90 and 270 deg
            object_normals->at(p).normal_x *= -1;
            object_normals->at(p).normal_y *= -1;
            object_normals->at(p).normal_z *= -1;
        }
    }

    pcl::concatenateFields(*object_cloud, *object_normals, *object_cloud); //data in the second cloud will overwrite the data in the first
    std::vector<int> nan;
    pcl::removeNaNNormalsFromPointCloud(*object_cloud, *object_cloud, nan);
}

//merge objects classified as NEW/REMOVED  with neighbouring objects classified as DISPLACED/STATIC
void ChangeDetection::mergeObjectParts(std::vector<DetectedObject> &detected_objects, std::string merge_object_parts_folder) {
    for (size_t i = 0; i < detected_objects.size(); i++) {
        if (detected_objects[i].state_ == ObjectState::NEW || detected_objects[i].state_ == ObjectState::REMOVED) {
            for (size_t j = 0; j < detected_objects.size(); j++) {
                if (detected_objects[j].state_ == ObjectState::STATIC || detected_objects[j].state_ == ObjectState::DISPLACED) {

                    if (detected_objects[j].getObjectCloud()->empty() || detected_objects[i].getObjectCloud()->empty())
                        continue;

                    pcl::PointCloud<PointNormal>::Ptr good_pts(new pcl::PointCloud<PointNormal>);
                    /// the detected objects is from the current frame
                    if (detected_objects[i].state_ == ObjectState::NEW && detected_objects[j].getID() == detected_objects[j].match_.object_id) {
                        /// are the potential overlapping objects close?
                        pcl::KdTreeFLANN<PointNormal> kdtree;
                        kdtree.setInputCloud (detected_objects[i].getObjectCloud());

                        bool is_obj_close = false;
                        int K = 1;
                        for (size_t p = 0; p < detected_objects[j].getObjectCloud()->size(); p++) {
                            const PointNormal &searchPoint = detected_objects[j].getObjectCloud()->points[p];
                            if (!pcl::isFinite(searchPoint))
                                continue;

                            std::vector<int> pointIdxKNNSearch(K);
                            std::vector<float> pointKNNSquaredDistance(K);

                            if ( kdtree.nearestKSearch (searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0 )
                            {
                                if (pointKNNSquaredDistance[0] < 0.02*0.02) {   //if any pair of points of the two objects is closer than 2 cm
                                    is_obj_close = true;
                                    break;
                                }
                            }
                        }
                        if (!is_obj_close)
                            continue;

                        pcl::ExtractIndices<PointNormal> extract;
                        extract.setInputCloud (detected_objects[j].getObjectCloud());
                        pcl::PointIndices::Ptr obj_ind(new pcl::PointIndices);
                        obj_ind->indices = detected_objects[j].match_.fitness_score.object_overlapping_pts;
                        extract.setIndices (obj_ind);
                        extract.setNegative (false);
                        extract.setKeepOrganized(false);
                        extract.filter(*good_pts);
                    }
                    /// the detected objects is from the reference frame
                    else if (detected_objects[i].state_ == ObjectState::REMOVED && detected_objects[j].getID() == detected_objects[j].match_.model_id) {
                        /// are the potential overlapping objects close?
                        pcl::KdTreeFLANN<PointNormal> kdtree;
                        kdtree.setInputCloud (detected_objects[i].getObjectCloud());

                        bool is_obj_close = false;
                        int K = 1;
                        for (size_t p = 0; p < detected_objects[j].getObjectCloud()->size(); p++) {
                            const PointNormal &searchPoint = detected_objects[j].getObjectCloud()->points[p];
                            if (!pcl::isFinite(searchPoint))
                                continue;

                            std::vector<int> pointIdxKNNSearch(K);
                            std::vector<float> pointKNNSquaredDistance(K);

                            if ( kdtree.nearestKSearch (searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0 )
                            {
                                if (pointKNNSquaredDistance[0] < 0.02*0.02) {   //if any pair of points of the two objects is closer than 2 cm
                                    is_obj_close = true;
                                    break;
                                }
                            }
                        }
                        if (!is_obj_close)
                            continue;

                        pcl::ExtractIndices<PointNormal> extract;
                        extract.setInputCloud (detected_objects[j].getObjectCloud());
                        pcl::PointIndices::Ptr obj_ind(new pcl::PointIndices);
                        obj_ind->indices = detected_objects[j].match_.fitness_score.model_overlapping_pts;
                        extract.setIndices (obj_ind);
                        extract.setNegative (false);
                        extract.setKeepOrganized(false);
                        extract.filter(*good_pts);
                    }
                    else
                        continue;

                    //call the region growing method
                    std::string path = merge_object_parts_folder + "/" + std::to_string(detected_objects[i].getID()) + "-" + std::to_string(detected_objects[j].getID());
                    boost::filesystem::create_directory(path);



                    pcl::PointCloud<PointNormal>::Ptr combined_object(new pcl::PointCloud<PointNormal>);
                    pcl::PointCloud<PointNormal>::Ptr object_i_refined(new pcl::PointCloud<PointNormal>);
                    pcl::PointCloud<PointNormal>::Ptr object_j_refined(new pcl::PointCloud<PointNormal>);
                    pcl::copyPointCloud(*detected_objects[i].getObjectCloud(), *object_i_refined);
                    pcl::copyPointCloud(*detected_objects[j].getObjectCloud(), *object_j_refined);
                    //refineNormals(object_i_refined);
                    //refineNormals(object_j_refined);

                    pcl::io::savePCDFileBinary(path + "/good_fitness_points.pcd", *good_pts);
                    pcl::io::savePCDFileBinary(path+ "/startingToMerge" + std::to_string(detected_objects[j].getID()) + ".pcd", *object_j_refined);
                    pcl::io::savePCDFileBinary(path+ "/tryToMerge" + std::to_string(detected_objects[i].getID()) + ".pcd", *object_i_refined);

                    //                    *combined_object += *detected_objects[i].getObjectCloud();
                    //                    *combined_object += *detected_objects[j].getObjectCloud();
                    *combined_object += *object_i_refined;
                    *combined_object += *object_j_refined;
                    pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
                    pcl::copyPointCloud(*combined_object, *scene_normals);
                    RegionGrowing<PointNormal, PointNormal> region_growing(combined_object, good_pts, scene_normals, false, 15.0, 10);
                    std::vector<int> add_object_ind = region_growing.compute();

                    //nothing was added to the static/displaced object
                    if (add_object_ind.size() == good_pts->size())
                        continue;


                    pcl::PointCloud<PointNormal>::Ptr extracted_cloud(new pcl::PointCloud<PointNormal>);
                    pcl::ExtractIndices<PointNormal> extract;
                    extract.setInputCloud (combined_object);
                    pcl::PointIndices::Ptr obj_ind(new pcl::PointIndices);
                    obj_ind->indices = add_object_ind;
                    extract.setIndices (obj_ind);
                    extract.setNegative (false);
                    extract.setKeepOrganized(false);
                    extract.filter(*extracted_cloud);
                    detected_objects[j].setObjectCloud(extracted_cloud);

                    extract.setNegative (true);
                    extract.setKeepOrganized(false);
                    extract.filter(*extracted_cloud);
                    detected_objects[i].setObjectCloud(extracted_cloud);


                    if (!detected_objects[j].getObjectCloud()->empty())
                        pcl::io::savePCDFileBinary(path + "/after_merging"+ std::to_string(detected_objects[j].getID()) + ".pcd", *detected_objects[j].getObjectCloud());
                    if (!detected_objects[i].getObjectCloud()->empty())
                        pcl::io::savePCDFileBinary(path + "/after_merging"+ std::to_string(detected_objects[i].getID()) + ".pcd", *detected_objects[i].getObjectCloud());

                    //remove very small clusters
                    std::vector<int> small_cluster_ind;
                    ObjectMatching::clusterOutliersBySize(detected_objects[i].getObjectCloud(), small_cluster_ind, 0.014, min_object_size_ds);

                    pcl::PointCloud<PointNormal>::Ptr small_cluster_cloud(new pcl::PointCloud<PointNormal>);
                    extract.setInputCloud (detected_objects[i].getObjectCloud());
                    obj_ind->indices = small_cluster_ind;
                    extract.setIndices (obj_ind);
                    extract.setNegative (false);
                    extract.setKeepOrganized(false);
                    extract.filter(*small_cluster_cloud);
                    *small_cluster_cloud += *detected_objects[j].getObjectCloud();
                    detected_objects[j].setObjectCloud(small_cluster_cloud);

                    extract.setNegative (true);
                    extract.setKeepOrganized(false);
                    extract.filter(*extracted_cloud);
                    detected_objects[i].setObjectCloud(extracted_cloud);

                    if (isObjectPlanar(detected_objects[i].getObjectCloud(), 0.01, 0.9)) {
                        pcl::PointCloud<PointNormal>::Ptr combined_cloud(new pcl::PointCloud<PointNormal>);
                        pcl::copyPointCloud(*detected_objects[j].getObjectCloud(), *combined_cloud);
                        *combined_cloud += *detected_objects[i].getObjectCloud();
                        detected_objects[j].setObjectCloud(combined_cloud);
                        detected_objects[i].clearClouds();
                    }

                    if (!detected_objects[j].getObjectCloud()->empty())
                        pcl::io::savePCDFileBinary(path + "/after_merging_cleaning"+ std::to_string(detected_objects[j].getID()) + ".pcd", *detected_objects[j].getObjectCloud());
                    if (!detected_objects[i].getObjectCloud()->empty())
                        pcl::io::savePCDFileBinary(path + "/after_merging_cleaning"+ std::to_string(detected_objects[i].getID()) + ".pcd", *detected_objects[i].getObjectCloud());
                }
            }
        }
    }

    //remove objects that have an empty cloud now
    detected_objects.erase(std::remove_if(
                               detected_objects.begin(),
                               detected_objects.end(),
                               [&](DetectedObject const & o) { return o.getObjectCloud()->size() == 0; }
                           ), detected_objects.end());
}




//TODO not only for one plane, but in the end of the whole pipeline!
//check clusters from new/removed objects if they could belong to a displaced/static cluster
void ChangeDetection::cleanResult(std::vector<DetectedObject> &detected_objects) {
    for (size_t i = 0; i < detected_objects.size(); i++) {
        if (detected_objects[i].state_ == ObjectState::NEW || detected_objects[i].state_ == ObjectState::REMOVED) {
            for (size_t j = 0; j < detected_objects.size(); j++) {
                if (detected_objects[j].state_ == ObjectState::STATIC || detected_objects[j].state_ == ObjectState::DISPLACED) {
                    //call the region growing method

                    if (detected_objects[j].getObjectCloud()->empty() || detected_objects[i].getObjectCloud()->empty())
                        continue;

                    //pcl::io::savePCDFileBinary("/home/edith/Desktop/before_rg_j.pcd", *detected_objects[j].getObjectCloud());
                    //pcl::io::savePCDFileBinary("/home/edith/Desktop/before_rg_i.pcd", *detected_objects[i].getObjectCloud());

                    pcl::PointCloud<PointNormal>::Ptr combined_object(new pcl::PointCloud<PointNormal>);
                    *combined_object += *detected_objects[i].getObjectCloud();
                    *combined_object += *detected_objects[j].getObjectCloud();
                    pcl::PointCloud<pcl::Normal>::Ptr obj_normals(new pcl::PointCloud<pcl::Normal>);
                    pcl::copyPointCloud(*combined_object, *obj_normals);
                    RegionGrowing<PointNormal, PointNormal> region_growing(combined_object, detected_objects[j].getObjectCloud(), obj_normals, false, 20.0);
                    std::vector<int> add_object_ind = region_growing.compute();

                    //nothing was added to the static/displaced object
                    if (add_object_ind.size() == detected_objects[j].getObjectCloud()->size())
                        continue;

                    //if less than 100 points remain, add everything to the static/displaced object
                    if (combined_object->size() - add_object_ind.size() < min_object_size_ds) {
                        detected_objects[j].setObjectCloud(combined_object);
                        detected_objects[i].setObjectCloud(boost::make_shared<pcl::PointCloud<PointNormal>>());
                    }

                    else {
                        pcl::PointCloud<PointNormal>::Ptr extracted_cloud(new pcl::PointCloud<PointNormal>);
                        pcl::ExtractIndices<PointNormal> extract;
                        extract.setInputCloud (combined_object);
                        pcl::PointIndices::Ptr obj_ind(new pcl::PointIndices);
                        obj_ind->indices = add_object_ind;
                        extract.setIndices (obj_ind);
                        extract.setNegative (false);
                        extract.setKeepOrganized(false);
                        extract.filter(*extracted_cloud);
                        detected_objects[j].setObjectCloud(extracted_cloud);

                        extract.setNegative (true);
                        extract.setKeepOrganized(false);
                        extract.filter(*extracted_cloud);
                        detected_objects[i].setObjectCloud(extracted_cloud);

                        //pcl::io::savePCDFileBinary("/home/edith/Desktop/after_rg_j.pcd", *detected_objects[j].getObjectCloud());
                        //pcl::io::savePCDFileBinary("/home/edith/Desktop/after_rg_i.pcd", *detected_objects[i].getObjectCloud());
                    }
                }
            }
        }
    }
}

//match extracted objects from one scene to the spatially close part of remaining points from the other scene
void ChangeDetection::matchAndRemoveObjects (pcl::PointCloud<PointNormal>::Ptr remaining_scene_points, pcl::PointCloud<PointNormal>::Ptr full_object_cloud,
                                             std::vector<PlaneWithObjInd> &extracted_objects) {
    for (std::vector<PlaneWithObjInd>::iterator obj_iter = extracted_objects.begin(); obj_iter != extracted_objects.end(); ) {
        //get object as cloud
        pcl::PointCloud<PointNormal>::Ptr object_cloud(new pcl::PointCloud<PointNormal>);
        pcl::ExtractIndices<PointNormal> extract;
        extract.setInputCloud (full_object_cloud);
        pcl::PointIndices::Ptr obj_ind(new pcl::PointIndices);
        obj_ind->indices = obj_iter -> obj_indices;
        extract.setIndices(obj_ind);
        extract.setNegative (false);
        extract.setKeepOrganized(false);
        extract.filter(*object_cloud);

        //get min and max values of object
        PointNormal minPt_object, maxPt_object;
        pcl::getMinMax3D (*object_cloud, minPt_object, maxPt_object);

        /// crop cloud
        float crop_margin = 0.5;
        pcl::PointCloud<PointNormal>::Ptr remaining_cloud_crop(new pcl::PointCloud<PointNormal>);
        pcl::PassThrough<PointNormal> pass;
        pass.setInputCloud(remaining_scene_points);
        pass.setFilterFieldName("x");
        pass.setFilterLimits(minPt_object.x-crop_margin, maxPt_object.x + crop_margin);
        pass.setKeepOrganized(false);
        pass.filter(*remaining_cloud_crop);
        pass.setInputCloud(remaining_cloud_crop);
        pass.setFilterFieldName("y");
        pass.setFilterLimits(minPt_object.y-crop_margin, maxPt_object.y + crop_margin);
        pass.setKeepOrganized(false);
        pass.filter(*remaining_cloud_crop);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(minPt_object.z-crop_margin, maxPt_object.z + crop_margin);
        pass.setKeepOrganized(false);
        pass.filter(*remaining_cloud_crop);

        if (remaining_cloud_crop->empty()) {
            ++obj_iter;
            continue;
        }

        pcl::io::savePCDFileBinary("/home/edith/Desktop/object.pcd", *object_cloud);
        pcl::io::savePCDFileBinary("/home/edith/Desktop/remaining_cloud_crop.pcd", *remaining_cloud_crop);

        //ICP alignment
        pcl::PointCloud<PointNormal>::Ptr object_registered(new pcl::PointCloud<PointNormal>());
        pcl::IterativeClosestPoint<PointNormal, PointNormal> icp;
        icp.setInputSource(object_cloud);
        icp.setInputTarget(remaining_cloud_crop);
        icp.setMaxCorrespondenceDistance(0.3);
        icp.setRANSACOutlierRejectionThreshold(icp_ransac_thr);
        icp.setMaximumIterations(icp_max_iter);
        icp.setTransformationEpsilon (1e-9);
        //icp.setTransformationRotationEpsilon(1 - 1e-15); //epsilon is the cos(angle)
        icp.align(*object_registered);



        pcl::io::savePCDFileBinary("/home/edith/Desktop/object_aligned.pcd", *object_registered);

        //check color
        v4r::apps::PPFRecognizerParameter params;
        FitnessScoreStruct fitness_score = ObjectMatching::computeModelFitness(object_registered, remaining_cloud_crop, params);
        if (fitness_score.object_conf > ppf_params.single_obj_min_fitness_weight_thr_) {
            obj_iter = extracted_objects.erase(obj_iter);
        }
        else {
            ++obj_iter;
        }
    }
}

DetectedObject ChangeDetection::fromPlaneIndObjToDetectedObject (pcl::PointCloud<PointNormal>::Ptr curr_cloud, PlaneWithObjInd obj) {
    pcl::PointCloud<PointNormal>::Ptr object_cloud(new pcl::PointCloud<PointNormal>);
    pcl::PointCloud<PointNormal>::Ptr plane_cloud(new pcl::PointCloud<PointNormal>);

    pcl::ExtractIndices<PointNormal> extract;
    extract.setInputCloud (curr_cloud);
    extract.setIndices (obj.plane.plane_ind);
    extract.setNegative (false);
    extract.setKeepOrganized(false);
    extract.filter(*plane_cloud);

    pcl::PointIndices::Ptr obj_ind(new pcl::PointIndices);
    obj_ind->indices = obj.obj_indices;
    extract.setIndices(obj_ind);
    extract.filter(*object_cloud);

    pcl::ModelCoefficients::Ptr plane_coeffs = boost::make_shared<pcl::ModelCoefficients>(*(obj.plane.coeffs));

    DetectedObject det_obj(object_cloud, plane_cloud, plane_coeffs);

    return det_obj;
}

void ChangeDetection::filterSmallVolumes(std::vector<DetectedObject> &objects, double volume_thr, int min_obj_size) {
    std::vector<int> filtered_ind;
    for (size_t i = 0;  i < objects.size(); i++) {
        if (objects[i].getObjectCloud()->size() < min_obj_size) {
            filtered_ind.push_back((i));
        } else {
            pcl::PointCloud<PointNormal>::Ptr cloud_hull (new pcl::PointCloud<PointNormal>);
            pcl::ConvexHull<PointNormal> chull;
            chull.setInputCloud (objects[i].getObjectCloudDS());
            chull.setDimension(3);
            chull.setComputeAreaVolume(true);
            chull.reconstruct (*cloud_hull);
            double volume = chull.getTotalVolume();
            if (volume < volume_thr) {
                //remove flat object
                std::cout << "Too little volume: " << objects[i].getID() << std::endl;
                filtered_ind.push_back((i));
            }
        }
    }
    for (int i = filtered_ind.size()-1; i >= 0; --i) {
        objects.erase(objects.begin() + filtered_ind[i]);
    }
}

std::vector<PlaneWithObjInd> ChangeDetection::getObjectsFromPlane(pcl::PointCloud<PointNormal>::Ptr input_cloud, Eigen::Vector4f plane_coeffs,
                                                                  pcl::PointCloud<pcl::PointXYZ>::Ptr convex_hull_pts,
                                                                  pcl::PointCloud<PointNormal>::Ptr prev_checked_plane_cloud, std::string res_path) {
    ExtractObjectsFromPlanes extract_curr_objects(input_cloud, plane_coeffs, convex_hull_pts,  res_path);
    std::vector<PlaneWithObjInd> objects_merged = extract_curr_objects.computeObjectsOnPlanes(prev_checked_plane_cloud);

    PointNormal nan_point;
    nan_point.x = nan_point.y = nan_point.z = std::numeric_limits<float>::quiet_NaN();

    pcl::PointCloud<PointNormal>::Ptr objects_plane_cloud (new pcl::PointCloud<PointNormal>(input_cloud->width, input_cloud->height, nan_point));
    std::vector<PlaneWithObjInd> objects;
    for (size_t c = 0; c < objects_merged.size(); c++) {
        pcl::PointCloud<PointNormal>::Ptr single_objects_plane_cloud (new pcl::PointCloud<PointNormal>(input_cloud->width, input_cloud->height, nan_point));
        const std::vector<int> &object_ind = objects_merged[c].obj_indices;
        for (size_t i = 0; i < object_ind.size(); i++) {
            single_objects_plane_cloud->points.at(object_ind[i]) = input_cloud->points.at(object_ind[i]);
        }
        std::vector<int> small_cluster_ind;
        std::vector<pcl::PointIndices> single_pot_object_ind = ObjectMatching::clusterOutliersBySize(single_objects_plane_cloud, small_cluster_ind, 0.02, 50);
        //cluster detected objects in separate objects
        for (size_t i = 0; i < single_pot_object_ind.size(); i++) {
            PlaneWithObjInd pot_object;
            pot_object.plane = objects_merged[c].plane;
            pot_object.obj_indices = single_pot_object_ind[i].indices;
            objects.push_back(pot_object);
        }
        for (size_t i = 0; i < single_pot_object_ind.size(); i++) {
            for (size_t j=0; j < single_pot_object_ind[i].indices.size(); j++) {
                objects_plane_cloud->points.at(single_pot_object_ind[i].indices[j]) = input_cloud->points.at(single_pot_object_ind[i].indices[j]);
            }
        }
        if (!objects_plane_cloud->empty())
            pcl::io::savePCDFileBinary(res_path + "/objects_from_plane.pcd", *objects_plane_cloud);

    }
    return objects;
}

//use the downsampled object cloud to filter objects with no volume, are planar or do not have enough points
void ChangeDetection::filterUnwantedObjects(std::vector<DetectedObject> &objects, double volume_thr, int min_obj_size, int max_obj_size,
                                            double plane_dist_thr, double plane_thr, std::string save_path) {
    std::vector<int> filtered_ind;
    for (size_t i = 0;  i < objects.size(); i++) {
        if (objects[i].state_ == ObjectState::DISPLACED || objects[i].state_ == ObjectState::STATIC)
            continue;

        if (save_path != "") {
            std::string obj_save_path = save_path + "/" + std::to_string(objects[i].getID());
            if (isObjectUnwanted(objects[i].getObjectCloudDS(), volume_thr, min_obj_size, max_obj_size, plane_dist_thr, plane_thr, obj_save_path))
                filtered_ind.push_back((i));
        }
        else if (isObjectUnwanted(objects[i].getObjectCloudDS(), volume_thr, min_obj_size, max_obj_size, plane_dist_thr, plane_thr))
            filtered_ind.push_back((i));
    }
    for (int i = filtered_ind.size()-1; i >= 0; --i) {
        objects.erase(objects.begin() + filtered_ind[i]);
    }
}

//uses the downsampled cloud to increase speed of ICP
//overwrites the original cloud!! --> upsampling afterwards needed
void ChangeDetection::performLV(std::vector<DetectedObject> &ref_objects, std::vector<DetectedObject> &curr_objects) {
    std::string LV_path =  output_path_ + "/LV_output/";
    boost::filesystem::create_directories(LV_path);

    std::vector<DetectedObject> curr_obj_static, ref_obj_static;
    int cnt = 0;

    LocalObjectVerificationParams lv_params{diff_dist, add_crop_static, icp_max_corr_dist, icp_max_corr_dist_plane,
                icp_max_iter, icp_ransac_thr, color_weight, overlap_weight, min_score_thr};

    for (std::vector<DetectedObject>::iterator co_it = curr_objects.begin(); co_it != curr_objects.end();) {
        bool erased_co_elem=false;
        pcl::PointCloud<PointNormal>::ConstPtr curr_obj_cloud = co_it->getObjectCloudDS();
        pcl::PointCloud<PointNormal>::ConstPtr curr_plane_cloud = co_it->plane_cloud_;

        pcl::KdTreeFLANN<PointNormal> kdtree;
        kdtree.setInputCloud (curr_obj_cloud);

        //find overlapping "objects" from the reference scene by finding corresponding planes first
        PointNormal minPt_object, maxPt_object;
        pcl::getMinMax3D (*curr_plane_cloud, minPt_object, maxPt_object);

        for (std::vector<DetectedObject>::iterator ro_it = ref_objects.begin(); ro_it != ref_objects.end();) {
            //check plane overlap
            pcl::PointCloud<PointNormal>::ConstPtr ref_plane_cloud = ro_it->plane_cloud_;

            /// crop plane
            float crop_margin = 0.1;
            pcl::PointCloud<PointNormal>::Ptr cloud_crop(new pcl::PointCloud<PointNormal>);
            pcl::PassThrough<PointNormal> pass;
            pass.setInputCloud(ref_plane_cloud);
            pass.setFilterFieldName("x");
            pass.setFilterLimits(minPt_object.x - crop_margin, maxPt_object.x + crop_margin);
            pass.setKeepOrganized(false);
            pass.filter(*cloud_crop);
            pass.setFilterFieldName("y");
            pass.setFilterLimits(minPt_object.y - crop_margin, maxPt_object.y + crop_margin);
            pass.filter(*cloud_crop);
            pass.setFilterFieldName("z");
            pass.setFilterLimits(minPt_object.z - crop_margin, maxPt_object.z + crop_margin);
            pass.filter(*cloud_crop);

            if (cloud_crop->empty()) {
                ro_it++;
                continue;
            }

            //else do LV
            //are the potential overlapping objects close?
            pcl::PointCloud<PointNormal>::ConstPtr ref_obj_cloud = ro_it->getObjectCloudDS();
            bool is_obj_close = false;
            int K = 1;
            for (size_t p = 0; p < ref_obj_cloud->size(); p++) {
                const PointNormal &searchPoint = ref_obj_cloud->points[p];
                if (!pcl::isFinite(searchPoint))
                    continue;

                std::vector<int> pointIdxKNNSearch(K);
                std::vector<float> pointKNNSquaredDistance(K);

                if ( kdtree.nearestKSearch (searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0 )
                {
                    if (pointKNNSquaredDistance[0] < 0.1*0.1) {   //if any pair of points of the two objects is closer than 10 cm --> LV
                        is_obj_close = true;
                        break;
                    }
                }
            }

            if (!is_obj_close) {
                ro_it++;
                continue;
            }
            else {
                LocalObjectVerification local_verification(ref_obj_cloud, curr_obj_cloud, lv_params);

                std::string LV_subfolder =  LV_path + "/" + std::to_string(cnt); cnt++;
                boost::filesystem::create_directories(LV_subfolder);
                local_verification.setDebugOutputPath(LV_subfolder);

                LVResult lv_result = local_verification.computeLV();

                //remove element in object vector
                if (lv_result.is_matched) {
                    //full match reference
                    if (lv_result.model_non_matching_pts.size() == 0) {
                        ro_it->state_ = ObjectState::STATIC;
                        ref_obj_static.push_back(*ro_it);
                        //ref_obj_static.push_back(*ro_it);
                        ro_it = ref_objects.erase(ro_it);
                    }
                    //extract matched  points if LV was successfull
                    else {
                        pcl::PointCloud<PointNormal>::Ptr matched_cloud(new pcl::PointCloud<PointNormal>);
                        pcl::ExtractIndices<PointNormal> extract;
                        extract.setInputCloud (ref_obj_cloud);
                        pcl::PointIndices::Ptr ind (new pcl::PointIndices);
                        ind->indices = lv_result.model_non_matching_pts;
                        extract.setIndices (ind);
                        extract.setKeepOrganized(false);
                        extract.setNegative (true);
                        extract.filter (*matched_cloud);

                        DetectedObject obj(matched_cloud, ro_it->plane_cloud_, ro_it->plane_coeffs_, ObjectState::STATIC);
                        ref_obj_static.push_back(obj);

                        pcl::PointCloud<PointNormal>::Ptr remaining_cloud(new pcl::PointCloud<PointNormal>);
                        extract.setNegative(false);
                        extract.filter (*remaining_cloud);

                        ro_it->setObjectCloud(remaining_cloud);
                        ro_it++;
                    }

                    if (lv_result.obj_non_matching_pts.size() == 0) {
                        co_it->state_ = ObjectState::STATIC;
                        curr_obj_static.push_back(*co_it);

                        //curr_obj_static.push_back(*co_it);
                        co_it = curr_objects.erase(co_it);
                        erased_co_elem = true;
                        //break; //if we break, the match will be not assigned to the objects
                    } else {
                        //extract matched  points
                        pcl::PointCloud<PointNormal>::Ptr matched_cloud(new pcl::PointCloud<PointNormal>);
                        pcl::ExtractIndices<PointNormal> extract;
                        extract.setInputCloud (curr_obj_cloud);
                        pcl::PointIndices::Ptr ind (new pcl::PointIndices);
                        ind->indices = lv_result.obj_non_matching_pts;
                        extract.setIndices (ind);
                        extract.setKeepOrganized(false);
                        extract.setNegative (true);
                        extract.filter (*matched_cloud);

                        DetectedObject obj(matched_cloud, co_it->plane_cloud_, co_it->plane_coeffs_, ObjectState::STATIC);
                        curr_obj_static.push_back(obj);

                        pcl::PointCloud<PointNormal>::Ptr remaining_cloud(new pcl::PointCloud<PointNormal>);
                        extract.setNegative(false);
                        extract.filter (*remaining_cloud);

                        co_it->setObjectCloud(remaining_cloud);
                        curr_obj_cloud = remaining_cloud;
                    }
                    /// associate the static objects
                    DetectedObject &matched_ref = ref_obj_static.back();
                    DetectedObject &matched_curr = curr_obj_static.back();

                    Match m(matched_ref.getID(), matched_curr.getID(), lv_result.transform_obj_to_model, lv_result.fitness_score);
                    matched_ref.match_ = m;
                    matched_curr.match_ = m;

                    std::string LV_match_path = LV_path + "/" + std::to_string(matched_ref.getID()) + "-" + std::to_string(matched_curr.getID());
                    boost::filesystem::create_directories(LV_match_path);
                    pcl::io::savePCDFileBinary(LV_match_path + "/ref_object.pcd", *matched_ref.getObjectCloud());
                    pcl::io::savePCDFileBinary(LV_match_path + "/curr_object.pcd", *matched_curr.getObjectCloud());

                    if (erased_co_elem)
                        break; //we found already an association, check next current element
                } else {
                    ro_it++;
                }
            }
        }
        if (!erased_co_elem) // && curr_objects.size() != 0) //otherwise the iterator goes out of scope
            co_it++;
    }

    ref_objects.insert(ref_objects.begin(), ref_obj_static.begin(), ref_obj_static.end());
    curr_objects.insert(curr_objects.begin(), curr_obj_static.begin(), curr_obj_static.end());
}

pcl::PointCloud<PointNormal>::Ptr ChangeDetection::fromDetObjectVecToCloud(const std::vector<DetectedObject> object_vec, bool withStaticObjects) {
    pcl::PointCloud<PointNormal>::Ptr object_cloud(new pcl::PointCloud<PointNormal>);
    for (DetectedObject obj : object_vec) {
        if (withStaticObjects || obj.state_ != ObjectState::STATIC)
            *object_cloud += *(obj.getObjectCloud());
    }
    return object_cloud;
}
