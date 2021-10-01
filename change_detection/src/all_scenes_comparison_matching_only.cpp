/**
  NOTES:
  -- if Eigen version < 3.3.6 comment line 277 and 278 in Eigen/src/Core/Matrix.h
        Base::_check_template_params();
        //if (RowsAtCompileTime!=Dynamic && ColsAtCompileTime!=Dynamic)
        //  Base::_set_noalias(other);
**/



#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <chrono>

#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>

#include "change_detection.h"

bool do_LV_before_matching = true;

typedef pcl::PointXYZRGBNormal PointNormal;
typedef pcl::PointXYZRGB PointRGB;
typedef pcl::PointXYZRGBL PointLabel;

typedef Eigen::Matrix<float,4,4,Eigen::DontAlign> Matrix4f_NotAligned;
typedef Eigen::Matrix<float,4,1,Eigen::DontAlign> Vector4f_NotAligned;

//storing single pcl points in a container leeds to weird alignment problems
struct Point3D {
    float x;
    float y;
    float z;

    static float squaredEuclideanDistance (const Point3D& p1, const Point3D& p2)
    {
        float diff_x = p2.x - p1.x, diff_y = p2.y - p1.y, diff_z = p2.z - p1.z;
        return (diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);
    }
};

struct ReconstructedPlane {
    pcl::PointCloud<PointNormal>::Ptr cloud;
    Point3D center_point;
    Vector4f_NotAligned plane_coeffs;
    pcl::PointCloud<pcl::PointXYZ>::Ptr convex_hull_cloud;
    bool is_checked;
};

std::map<int, DetectedObject> new_obj;
std::map<int, DetectedObject> removed_obj;
std::map<int, DetectedObject> pot_new_obj;
std::map<int, DetectedObject> pot_removed_obj;
std::map<int, DetectedObject> ref_displaced_obj;
std::map<int, DetectedObject> curr_displaced_obj;
std::map<int, DetectedObject> curr_static_obj;
std::map<int, DetectedObject> ref_static_obj;

std::string ppf_model_path;
std::string base_result_path;
std::string result_path;


std::vector<DetectedObject> fromMapToValVec(std::map<int, DetectedObject> map) {
    //transform map into vec to be able to call object matching
    std::vector<DetectedObject> vec;
    vec.reserve(map.size());
    for(auto const& imap: map)
        vec.push_back(imap.second);
    return vec;
}

template <typename DetectedObject>
ostream& operator<<(ostream& output, std::map<int, DetectedObject> const& values)
{
    for (auto const & value : values)
    {
        const int &id =  value.first;
        output << id << " ";
    }
    return output;
}

std::string getCurrentTime() {
    time_t rawtime = time(nullptr);
    struct tm * now = localtime( & rawtime );
    char buffer [80];
    strftime (buffer,80,"%Y-%m-%d-%H:%M",now);
    return std::string(buffer);
}

bool readInput(std::string input_path, const pcl::PointCloud<PointNormal>::Ptr &cloud) {
    std::string ext = input_path.substr(input_path.length()-3, input_path.length());
    if (ext == "ply") {
        if (pcl::io::loadPLYFile<PointNormal> (input_path, *cloud) == -1)
        {
            std::cerr << "Couldn't read file " << input_path << std::endl;
            return false;
        }
        return true;
    } else if (ext == "pcd") {
        if (pcl::io::loadPCDFile<PointNormal> (input_path, *cloud) == -1)
        {
            std::cerr << "Couldn't read file " << input_path << std::endl;
            return false;
        }
        return true;
    } else {
        std::cerr << "Couldn't read file " << input_path << ". It needs to be a ply or pcd file" << std::endl;
        return false;
    }
}

void createNewModelFolder(DetectedObject &ro, std::string ppf_model_path, std::string result_path) {
    std::string orig_path = ppf_model_path + "/" + std::to_string(ro.getID());
    ro.object_folder_path_ = orig_path;

    boost::filesystem::create_directories(orig_path);
    pcl::io::savePCDFile(orig_path + "/3D_model.pcd", *ro.getObjectCloud()); //PPF will create a new model with the new cloud
}


void removeModelFolder(DetectedObject ro, std::string ppf_model_path, std::string result_path) {
    //there should already exist a folder
    std::string orig_path = ppf_model_path + "/" + std::to_string(ro.getID());
    if (boost::filesystem::exists(orig_path)) {
        boost::filesystem::path dest_folder = result_path + "/model_partially_matched/" + std::to_string(ro.getID());
        boost::filesystem::create_directories(dest_folder);
        //count pcd-files in the folder
        int nr_pcd_files = 0;
        boost::filesystem::directory_iterator end_iter; // Default constructor for an iterator is the end iterator
        for (boost::filesystem::directory_iterator iter(dest_folder); iter != end_iter; ++iter) {
            if (iter->path().extension() == ".pcd")
                ++nr_pcd_files;
        }

        //copy the folder to another directory. The model is already matched and should not be used anymore
        for (const auto& dirEnt : boost::filesystem::recursive_directory_iterator{orig_path})
        {
            if (dirEnt.path().extension() == ".pcd") {
                const auto& path = dirEnt.path();
                boost::filesystem::path dest_filename = path.stem().string() + std::to_string(nr_pcd_files) + path.extension().string();
                boost::filesystem::copy(path, dest_folder / dest_filename);
                std::cout << "Copy " << path << " to " << (dest_folder / path.filename()) << std::endl;
            }
        }
    }
    boost::filesystem::remove_all(orig_path);
}

void updateDetectedObjects(std::vector<DetectedObject>& ref_result, std::vector<DetectedObject>& curr_result) {
    for (DetectedObject ro : ref_result) {
        if (ro.state_ == ObjectState::REMOVED) {
            //means that there was a partial match and have to create a new model folder
            if (ro.object_folder_path_ == "") {
                createNewModelFolder(ro, ppf_model_path, result_path);
            }
            pot_removed_obj[ro.getID()] = ro;
        } else if (ro.state_ == ObjectState::DISPLACED) {
            //means that there was a partial match and we do not need the model anymore
            if (ro.object_folder_path_ == "") {
                removeModelFolder(ro, ppf_model_path, result_path);
            }
            ref_displaced_obj[ro.getID()] = ro;
            pot_removed_obj.erase(ro.getID());
        } else if (ro.state_ == ObjectState::STATIC) {
            //means that there was a partial match and we do not need the model anymore
            if (ro.object_folder_path_ == "") {
                removeModelFolder(ro, ppf_model_path, result_path);
            }
            ref_static_obj[ro.getID()] = ro;
            pot_removed_obj.erase(ro.getID());
        }

    }
    for (DetectedObject co : curr_result) {
        if (co.state_ == ObjectState::NEW) {
            pot_new_obj[co.getID()] = co;
        } else if (co.state_ == ObjectState::DISPLACED) {
            curr_displaced_obj[co.getID()] = co;
            pot_new_obj.erase(co.getID());
        }  else if (co.state_ == ObjectState::STATIC) {
            curr_static_obj[co.getID()] = co;
            pot_new_obj.erase(co.getID());
        }
    }
}



int main(int argc, char* argv[])
{
    /// Check arguments and print info
    if (argc < 4) {
        pcl::console::print_info("\n\
                                 -- Object change detection based on reconstructions of a plane of interest at two different timestamps for all scenes in the given folder -- : \n\
                                 \n\
                                 Syntax: %s room_path \n\
                                 [Options] \n\
                                 -r path, where results should be stored, a folder with date and time gets created there \n\
                                 -c config path for ppf params",
                                 argv[0]);
        return(1);
    }

    /// Parse command line arguments
    std::string room_path = argv[1];
    std::string ppf_config_path="";
    pcl::console::parse(argc, argv, "-r", base_result_path);
    pcl::console::parse(argc, argv, "-c", ppf_config_path);

    //extract all scene folders
    if (!boost::filesystem::exists(room_path) || !boost::filesystem::is_directory(room_path)) {
        std::cout << room_path << " does not exist or is not a directory" << std::endl;
        return -1;
    }

    std::vector<std::string> all_scene_paths;
    const std::regex scene_filter("scene[2-9]+-scene[2-9]+");
    for(auto & scene_path : boost::filesystem::directory_iterator(room_path))
    {
        if (boost::filesystem::is_directory(scene_path.status()))
        {
            if(std::regex_match(scene_path.path().filename().string(), scene_filter ) ) {
                all_scene_paths.push_back(scene_path.path().filename().string());
            }
        }
    }


    std::string timestamp = getCurrentTime();
    base_result_path =  base_result_path + "/" + timestamp + (do_LV_before_matching ? "_withLV":"" ) + "_filterPlanarObj_ds05/";
    //----------------------------setup result folder----------------------------------

    for (size_t i = 0; i < all_scene_paths.size(); i++) {
        result_path = base_result_path + all_scene_paths[i] + "/";
        boost::filesystem::create_directories(result_path);

        boost::filesystem::copy(ppf_config_path, result_path+"/config.ini");

        //----------------------------setup ppf model folder-------------------------------
        ppf_model_path = result_path + "/model_objects/";
        boost::filesystem::create_directories(ppf_model_path);

        pot_removed_obj.clear();
        pot_new_obj.clear();
        ref_displaced_obj.clear();
        curr_displaced_obj.clear();
        ref_static_obj.clear();
        curr_static_obj.clear();
        new_obj.clear();
        removed_obj.clear();


        //iterate through all plane comparisons
        std::string absolute_scene_comp_path = room_path + "/" + all_scene_paths[i] + "/";
        const std::regex plane_comp_filter("[0-9]*_[0-9]*");
        for(auto & plane_comp_path : boost::filesystem::directory_iterator(absolute_scene_comp_path))
        {
            if (boost::filesystem::is_directory(plane_comp_path.status()))
            {
                if(std::regex_match(plane_comp_path.path().filename().string(), plane_comp_filter ) ) { //ref-plane_curr_plane
                    std::cout << plane_comp_path << std::endl;

                    std::string merge_object_parts_folder = result_path + "/" + plane_comp_path.path().filename().string()  + "/mergeObjectParts";
                    boost::filesystem::create_directories(merge_object_parts_folder);

                    //read ref and curr objects from folders
                    std::string ref_objs_path = plane_comp_path.path().string() + "/model_objects/";
                    std::string curr_objs_path = plane_comp_path.path().string() + "/current_det_objects/";

                    std::vector<DetectedObject> ref_objects_vec, ref_result;
                    std::vector<DetectedObject> curr_objects_vec, curr_result;

                    if (boost::filesystem::exists(ref_objs_path)) {
                        for(auto & model_path : boost::filesystem::directory_iterator(ref_objs_path))
                        {
                            if (boost::filesystem::is_directory(model_path.status()))
                            {
                                pcl::PointCloud<PointNormal>::Ptr obj_cloud (new pcl::PointCloud<PointNormal>);
                                pcl::PointCloud<PointNormal>::Ptr plane_cloud (new pcl::PointCloud<PointNormal>);
                                if (!readInput(model_path.path().string() + "/3D_model.pcd", obj_cloud))
                                    return false;
                                if (!readInput(model_path.path().string() + "/plane.pcd", plane_cloud))
                                    return false;
                                if (obj_cloud->size() > 3000 || obj_cloud->size() < 200)
                                    return false;
                                DetectedObject model_obj(obj_cloud, plane_cloud);
                                std::string obj_folder = ppf_model_path + std::to_string(model_obj.getID()); //PPF uses the folder name as model_id!
                                boost::filesystem::create_directories(obj_folder);
                                pcl::io::savePCDFile(obj_folder + "/3D_model.pcd", *(model_obj.getObjectCloud()));
                                model_obj.object_folder_path_ = obj_folder;
                                ref_objects_vec.push_back(model_obj);
                            }
                        }
                    }

                    if (boost::filesystem::exists(curr_objs_path)) {
                        for(auto & obj_path : boost::filesystem::directory_iterator(curr_objs_path))
                        {
                            if (boost::filesystem::is_directory(obj_path.status()))
                            {
                                std::cout << obj_path << std::endl;
                                pcl::PointCloud<PointNormal>::Ptr obj_cloud (new pcl::PointCloud<PointNormal>);
                                pcl::PointCloud<PointNormal>::Ptr plane_cloud (new pcl::PointCloud<PointNormal>);
                                if (!readInput(obj_path.path().string() + "/object.pcd", obj_cloud))
                                    return false;
                                if (!readInput(obj_path.path().string() + "/plane.pcd", plane_cloud))
                                    return false;
                                if (obj_cloud->size() > 3000 || obj_cloud->size() < 200)
                                    return false;
                                DetectedObject obj(obj_cloud, plane_cloud);
                                curr_objects_vec.push_back(obj);
                            }
                        }
                    }

                    ChangeDetection::filterSmallVolumes(ref_objects_vec, min_object_volume, min_object_size);
                    ChangeDetection::filterSmallVolumes(curr_objects_vec, min_object_volume, min_object_size);

                    if (curr_objects_vec.size() == 0) { //all objects removed from ref scene
                        for (size_t i = 0; i < ref_objects_vec.size(); i++) {
                            ref_objects_vec[i].state_ = ObjectState::REMOVED;
                        }
                        ref_result = ref_objects_vec;
                        updateDetectedObjects(ref_result, curr_result);
                        continue;
                    }

                    if (ref_objects_vec.size() == 0) { //all objects are new in curr
                        for (size_t i = 0; i < curr_objects_vec.size(); i++) {
                            curr_objects_vec[i].state_ = ObjectState::NEW;
                        }
                        curr_result = curr_objects_vec;
                        updateDetectedObjects(ref_result, curr_result);
                        continue;
                    }

                    //matches between same plane different timestamps
                    ObjectMatching object_matching(ref_objects_vec, curr_objects_vec, ppf_model_path, ppf_config_path);
                    std::vector<Match> matches = object_matching.compute(ref_result, curr_result);

                    //region growing of static/displaced objects (should create more precise results if e.g. the model was smaller than die object or not precisely aligned
                    ChangeDetection::mergeObjectParts(ref_result, merge_object_parts_folder);
                    ChangeDetection::mergeObjectParts(curr_result, merge_object_parts_folder);
                    ChangeDetection::filterSmallVolumes(ref_result, min_object_volume, min_object_size);
                    ChangeDetection::filterSmallVolumes(curr_result, min_object_volume, min_object_size);

                    //all detected objects labeled as removed (ref_objects) or new (curr_objects) could be placed on another plane
                    updateDetectedObjects(ref_result, curr_result);
                }

                if(plane_comp_path.path().filename().string().rfind("curr_",0) == 0 ) { //ref-plane_curr_plane
                    std::cout << plane_comp_path << std::endl;

                    std::string merge_object_parts_folder = result_path + "/" + plane_comp_path.path().filename().string()  + "/mergeObjectParts";
                    boost::filesystem::create_directory(merge_object_parts_folder);

                    std::string curr_objs_path = plane_comp_path.path().string() + "/current_det_objects/";

                    if (boost::filesystem::exists(curr_objs_path)) {
                        std::vector<DetectedObject> curr_objects_vec;

                        for(auto & obj_path : boost::filesystem::directory_iterator(curr_objs_path))
                        {
                            if (boost::filesystem::is_directory(obj_path.status()))
                            {
                                pcl::PointCloud<PointNormal>::Ptr obj_cloud (new pcl::PointCloud<PointNormal>);
                                pcl::PointCloud<PointNormal>::Ptr plane_cloud (new pcl::PointCloud<PointNormal>);
                                if (!readInput(obj_path.path().string() + "/object.pcd", obj_cloud))
                                    return false;
                                if (!readInput(obj_path.path().string() + "/plane.pcd", plane_cloud))
                                    return false;
                                if (obj_cloud->size() > 3000 || obj_cloud->size() < 200)
                                    return false;
                                DetectedObject obj (obj_cloud, plane_cloud);
                                curr_objects_vec.push_back(obj);
                            }
                        }
                        ChangeDetection::filterSmallVolumes(curr_objects_vec, min_object_volume, min_object_size);

                        std::vector<DetectedObject> ref_result, curr_result;
                        curr_result = curr_objects_vec;
                        updateDetectedObjects(ref_result, curr_result);
                    }
                }

                if(plane_comp_path.path().filename().string().rfind("ref_",0) == 0 ) { //ref-plane_curr_plane
                    std::cout << plane_comp_path << std::endl;

                    std::string merge_object_parts_folder = result_path + "/" + plane_comp_path.path().filename().string()  + "/mergeObjectParts";
                    boost::filesystem::create_directory(merge_object_parts_folder);

                    std::string ref_objs_path = plane_comp_path.path().string() + "/model_objects/";

                    if (boost::filesystem::exists(ref_objs_path)) {
                        std::vector<DetectedObject> ref_objects_vec;

                        for(auto & obj_path : boost::filesystem::directory_iterator(ref_objs_path))
                        {
                            if (boost::filesystem::is_directory(obj_path.status()))
                            {
                                pcl::PointCloud<PointNormal>::Ptr obj_cloud (new pcl::PointCloud<PointNormal>);
                                pcl::PointCloud<PointNormal>::Ptr plane_cloud (new pcl::PointCloud<PointNormal>);
                                if (!readInput(obj_path.path().string() + "/3D_model.pcd", obj_cloud))
                                    return false;
                                if (!readInput(obj_path.path().string() + "/plane.pcd", plane_cloud))
                                    return false;
                                if (obj_cloud->size() > 3000 || obj_cloud->size() < 200)
                                    return false;
                                DetectedObject obj (obj_cloud, plane_cloud);
                                ref_objects_vec.push_back(obj);
                            }
                        }
                        ChangeDetection::filterSmallVolumes(ref_objects_vec, min_object_volume, min_object_size);

                        std::vector<DetectedObject> ref_result, curr_result;
                        ref_result = ref_objects_vec;
                        updateDetectedObjects(ref_result, curr_result);
                    }
                }
            }
        }

        //after collecting potential new and removed objects from the plane, try to match them
        if (pot_removed_obj.size() != 0 && pot_new_obj.size() != 0) {
            std::string merge_object_parts_folder = result_path + "/leftover_mergeObjectParts";
            boost::filesystem::create_directory(merge_object_parts_folder);

            //transform map into vec to be able to call object matching
            std::vector<DetectedObject> pot_rem_obj_vec, pot_new_obj_vec;
            pot_rem_obj_vec = fromMapToValVec(pot_removed_obj);
            pot_new_obj_vec = fromMapToValVec(pot_new_obj);
            ObjectMatching matching(pot_rem_obj_vec, pot_new_obj_vec, ppf_model_path, ppf_config_path);
            std::vector<DetectedObject> ref_result, curr_result;
            matching.compute(ref_result, curr_result);

            ChangeDetection::mergeObjectParts(ref_result, merge_object_parts_folder);
            ChangeDetection::mergeObjectParts(curr_result, merge_object_parts_folder);

            ChangeDetection::filterSmallVolumes(ref_result, min_object_volume, min_object_size);
            ChangeDetection::filterSmallVolumes(curr_result, min_object_volume, min_object_size);

            updateDetectedObjects(ref_result, curr_result);
        }


        //all pot. moved objects are in the end either removed or new
        removed_obj = pot_removed_obj;
        new_obj = pot_new_obj;

        std::cout << "FINAL RESULT" << std::endl;
        std::cout << "Removed objects: " << removed_obj << std::endl;
        std::cout << "New objects: " << new_obj << std::endl;
        std::cout << "Displaced objects in reference: " << ref_displaced_obj << std::endl;
        std::cout << "Displaced objects in current: " << curr_displaced_obj << std::endl;
        std::cout << "Static objects in reference: " << ref_static_obj << std::endl;
        std::cout << "Static objects in current: " << curr_static_obj << std::endl;


        //STORING RESULTS AND VISUALIZE THEM

        //create point clouds of detected objects to save results as pcd-files
        pcl::PointCloud<PointNormal>::Ptr ref_removed_objects_cloud(new pcl::PointCloud<PointNormal>);
        pcl::PointCloud<PointLabel>::Ptr ref_displaced_objects_cloud(new pcl::PointCloud<PointLabel>);
        pcl::PointCloud<PointLabel>::Ptr ref_static_objects_cloud(new pcl::PointCloud<PointLabel>);
        pcl::PointCloud<PointNormal>::Ptr curr_new_objects_cloud(new pcl::PointCloud<PointNormal>);
        pcl::PointCloud<PointLabel>::Ptr curr_displaced_objects_cloud(new pcl::PointCloud<PointLabel>);
        pcl::PointCloud<PointLabel>::Ptr curr_static_objects_cloud(new pcl::PointCloud<PointLabel>);

        //transform maps to vectors
        std::vector<DetectedObject> removed_obj_vec, new_obj_vec, ref_dis_obj_vec, curr_dis_obj_vec, ref_static_obj_vec, curr_static_obj_vec;
        removed_obj_vec = fromMapToValVec(removed_obj);
        new_obj_vec = fromMapToValVec(new_obj);
        ref_dis_obj_vec = fromMapToValVec(ref_displaced_obj);
        curr_dis_obj_vec = fromMapToValVec(curr_displaced_obj);
        ref_static_obj_vec = fromMapToValVec(ref_static_obj);
        curr_static_obj_vec = fromMapToValVec(curr_static_obj);

        for (auto const & o : removed_obj) {
            *ref_removed_objects_cloud += *(o.second.getObjectCloud());
        }
        if (!ref_removed_objects_cloud->empty())
            pcl::io::savePCDFile(result_path + "/ref_removed_objects.pcd", *ref_removed_objects_cloud);

        for (auto const & o : new_obj) {
            *curr_new_objects_cloud += *(o.second.getObjectCloud());
        }
        if (!curr_new_objects_cloud->empty())
            pcl::io::savePCDFile(result_path + "/curr_new_objects.pcd", *curr_new_objects_cloud);

        //assign labels to the object based on the matches for DISPLACED objects
        for (size_t o = 0; o < ref_dis_obj_vec.size(); o++) {
            const DetectedObject &ref_object = ref_dis_obj_vec[o];
            auto curr_obj_iter = std::find_if( curr_dis_obj_vec.begin(), curr_dis_obj_vec.end(),[ref_object]
                                               (DetectedObject const &o) {return o.match_.model_id == ref_object.getID(); });
            const DetectedObject &curr_object = *curr_obj_iter;
            pcl::PointCloud<PointLabel>::Ptr ref_objects_cloud(new pcl::PointCloud<PointLabel>);
            pcl::PointCloud<PointLabel>::Ptr curr_objects_cloud(new pcl::PointCloud<PointLabel>);
            pcl::copyPointCloud(*ref_object.getObjectCloud(), *ref_objects_cloud);
            pcl::copyPointCloud(*curr_object.getObjectCloud(), *curr_objects_cloud);
            for (size_t i = 0; i < ref_objects_cloud->size(); i++) {
                ref_objects_cloud->points[i].label=ref_object.getID() * 20;
            }
            for (size_t i = 0; i < curr_objects_cloud->size(); i++) {
                curr_objects_cloud->points[i].label = ref_object.getID() * 20;
            }
            pcl::io::savePCDFileBinary("/home/edith/Desktop/curr_object_displ.pcd", *curr_objects_cloud);
            *ref_displaced_objects_cloud += *ref_objects_cloud;
            *curr_displaced_objects_cloud += *curr_objects_cloud;
        }

        if (!ref_displaced_objects_cloud->empty())
            pcl::io::savePCDFile(result_path + "/ref_displaced_objects.pcd", *ref_displaced_objects_cloud);
        if (!curr_displaced_objects_cloud->empty())
            pcl::io::savePCDFile(result_path + "/curr_displaced_objects.pcd", *curr_displaced_objects_cloud);


        //assign labels to the object based on the matches for STATIC objects
        for (size_t o = 0; o < ref_static_obj_vec.size(); o++) {
            const DetectedObject &ref_object = ref_static_obj_vec[o];
            auto curr_obj_iter = std::find_if( curr_static_obj_vec.begin(), curr_static_obj_vec.end(),[ref_object]
                                               (DetectedObject const &o) {return o.match_.model_id == ref_object.getID(); });
            const DetectedObject &curr_object = *curr_obj_iter;
            pcl::PointCloud<PointLabel>::Ptr ref_objects_cloud(new pcl::PointCloud<PointLabel>);
            pcl::PointCloud<PointLabel>::Ptr curr_objects_cloud(new pcl::PointCloud<PointLabel>);
            pcl::copyPointCloud(*ref_object.getObjectCloud(), *ref_objects_cloud);
            pcl::copyPointCloud(*curr_object.getObjectCloud(), *curr_objects_cloud);
            for (size_t i = 0; i < ref_objects_cloud->size(); i++) {
                ref_objects_cloud->points[i].label = ref_object.getID() * 20;
            }
            for (size_t i = 0; i < curr_objects_cloud->size(); i++) {
                curr_objects_cloud->points[i].label = ref_object.getID() * 20;
            }
            *ref_static_objects_cloud += *ref_objects_cloud;
            *curr_static_objects_cloud += *curr_objects_cloud;
        }
        if (!ref_static_objects_cloud->empty())
            pcl::io::savePCDFile(result_path + "/ref_static_objects.pcd", *ref_static_objects_cloud);
        if (!curr_static_objects_cloud->empty())
            pcl::io::savePCDFile(result_path + "/curr_static_objects.pcd", *curr_static_objects_cloud);

    }
}


