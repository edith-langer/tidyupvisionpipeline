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

//map of map because for one plane several occurrences are possible
std::map<int, std::map<int, Matrix4f_NotAligned>> transformationParser(std::string path) {
    std::map<int, std::map<int, Matrix4f_NotAligned>> transformation_map;
    std::ifstream file(path);
    if (!file.good()) {
        std::cerr << "File " << path << "does not exist << std::endl";
        return transformation_map;
    }
    std::string str;
    //Table 1, start at 1567875713.42, occurance 0. Transform:
    while (std::getline(file, str)) {
        if (str.rfind("Table",0) == 0) {
            std::vector<std::string> row_split_result;
            boost::split(row_split_result, str, boost::is_any_of(",")); //split the first row
            std::vector<std::string> split_result;
            boost::split(split_result, row_split_result[0], boost::is_any_of(" ")); //split "Table 1"
            int table_nr = std::stoi(split_result[1]);
            boost::split(split_result, row_split_result[2], boost::is_any_of(" ")); //split " occurance 1. Transform"
            split_result[2].pop_back();
            int occurrance = std::stoi(split_result[2]);

            //now extract the transformation matrix
            Matrix4f_NotAligned mat;
            std::getline(file, str);
            boost::split(split_result, str, boost::is_any_of("[[ ]"), boost::token_compress_on);
            mat(0,0) = std::stof(split_result[2]);
            mat(0,1) = std::stof(split_result[3]);
            mat(0,2) = std::stof(split_result[4]);
            mat(0,3) = std::stof(split_result[5]);

            std::getline(file, str);
            str = str.substr(2, str.size() - 3); //remove the first 2 chars and the last one
            boost::trim(str);
            std::vector<std::string> transf_split_result;
            boost::split(transf_split_result, str, boost::is_any_of(" "), boost::token_compress_on);
            mat(1,0) = std::stof(transf_split_result[0]);
            mat(1,1) = std::stof(transf_split_result[1]);
            mat(1,2) = std::stof(transf_split_result[2]);
            mat(1,3) = std::stof(transf_split_result[3]);

            std::getline(file, str);
            str = str.substr(2, str.size() - 3); //remove the first 2 chars and the last one
            boost::trim(str);
            boost::split(transf_split_result, str, boost::is_any_of(" "), boost::token_compress_on);
            mat(2,0) = std::stof(transf_split_result[0]);
            mat(2,1) = std::stof(transf_split_result[1]);
            mat(2,2) = std::stof(transf_split_result[2]);
            mat(2,3) = std::stof(transf_split_result[3]);

            std::getline(file, str);
            str = str.substr(2, str.size() - 4); //remove the first 2 chars and the last two
            boost::trim(str);
            boost::split(transf_split_result, str, boost::is_any_of(" "), boost::token_compress_on);
            mat(3,0) = std::stof(transf_split_result[0]);
            mat(3,1) = std::stof(transf_split_result[1]);
            mat(3,2) = std::stof(transf_split_result[2]);
            mat(3,3) = std::stof(transf_split_result[3]);

            //transformation_map[table_nr].insert(std::make_pair(occurrance, mat.inverse())); //from camera frame to map frame
            transformation_map[table_nr].insert(std::make_pair(occurrance, Eigen::Matrix4f::Identity()));
        }
    }
    return transformation_map;
}

std::map<int, ReconstructedPlane> convexHullPtsParser(std::string path) {
    std::map<int, ReconstructedPlane> rec_planes;
    std::ifstream file(path);
    if (!file.good()) {
        std::cerr << "File " << path << "does not exist << std::endl";
        return rec_planes;
    }
    std::string str;
    int plane_cnt = 0;
    while (std::getline(file, str)) {
        if (str.rfind("center",0) == 0) {
            std::getline(file, str);
            Point3D point;
            std::vector<std::string> split_str;
            boost::split(split_str, str, boost::is_any_of(":"));
            point.x = std::stof(split_str[1]);

            std::getline(file, str);
            boost::split(split_str, str, boost::is_any_of(":"));
            point.y = std::stof(split_str[1]);

            std::getline(file, str);
            boost::split(split_str, str, boost::is_any_of(":"));
            point.z = std::stof(split_str[1]);

            rec_planes[plane_cnt].center_point = point;
            continue;
        }
        if (str.rfind("points",0) != 0) {
            continue;
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        while (std::getline(file, str) && str.rfind("plane", 0) != 0) {
            boost::trim(str);
            if (str.rfind("x",0) == 0) {
                pcl::PointXYZ point;
                std::vector<std::string> split_str;
                boost::split(split_str, str, boost::is_any_of(":"));
                point.x = std::stof(split_str[1]);

                std::getline(file, str);
                boost::split(split_str, str, boost::is_any_of(":"));
                point.y = std::stof(split_str[1]);

                std::getline(file, str);
                boost::split(split_str, str, boost::is_any_of(":"));
                point.z = std::stof(split_str[1]);

                cloud->points.push_back(point);
            }
        }
        cloud->height=1;
        cloud->width=cloud->points.size();
        rec_planes[plane_cnt].convex_hull_cloud = cloud;

        //now it is time to parse plane:
        std::getline(file, str);
        Vector4f_NotAligned plane_coeffs;
        std::vector<std::string> split_str;
        boost::split(split_str, str, boost::is_any_of(":"));
        plane_coeffs[0] = std::stof(split_str[1]); //x

        std::getline(file, str);
        boost::split(split_str, str, boost::is_any_of(":"));
        plane_coeffs[1] = std::stof(split_str[1]); //y

        std::getline(file, str);
        boost::split(split_str, str, boost::is_any_of(":"));
        plane_coeffs[2] = std::stof(split_str[1]); //z

        std::getline(file, str);
        boost::split(split_str, str, boost::is_any_of(":"));
        plane_coeffs[3] = std::stof(split_str[1]); //d

        rec_planes[plane_cnt].plane_coeffs = plane_coeffs;

        plane_cnt += 1;
    }
    return rec_planes;
}

std::map<int, ReconstructedPlane> prepareInputData(std::string data_path) {
    std::map<int, ReconstructedPlane> rec_planes;
    if(!boost::filesystem::exists(data_path) && !boost::filesystem::is_directory(data_path) ) {
        std::cerr << "The data path does not exist or is not a directory " << data_path << std::endl;
        return rec_planes;
    }

    //TODO Read plane coeffs and convex hull points from DB
    std::map<int, ReconstructedPlane> center_and_convexHull= convexHullPtsParser(data_path + "/table.txt");
    std::map<int, std::map<int, Matrix4f_NotAligned>> transformations = transformationParser(data_path + "/log.txt");

    //iterate through all plane folders and extract the reconstructed ply-files
    int plane_nr = 0;
    bool found_plane_folder = true;
    while (found_plane_folder) {
        std::string plane_nr_path = data_path + "/planes/" + std::to_string(plane_nr);
        if (boost::filesystem::exists(plane_nr_path) && boost::filesystem::is_directory(plane_nr_path)) {
            pcl::PointCloud<PointNormal>::Ptr plane_cloud(new pcl::PointCloud<PointNormal>);
            if (!readInput(plane_nr_path + "/merged_plane_clouds_ds002.pcd", plane_cloud)) {
                continue;
            }
            //transform the ply
            Matrix4f_NotAligned & mat = transformations[plane_nr][0];
            pcl::transformPointCloudWithNormals(*plane_cloud, *plane_cloud, mat);

            //save in vector of planeStructs
            ReconstructedPlane rec_plane;
            rec_plane.cloud = plane_cloud;
            rec_plane.is_checked = false;
            rec_planes[plane_nr] = rec_plane;
            plane_nr++;
        } else {
            found_plane_folder = false;
        }
    }

    //merge center point and convex hull points with loaded point cloud
    for (std::map<int, ReconstructedPlane>::iterator it = center_and_convexHull.begin(); it != center_and_convexHull.end(); it++) {
        rec_planes.at(it->first).center_point = it->second.center_point;
        rec_planes.at(it->first).convex_hull_cloud = it->second.convex_hull_cloud;
        rec_planes.at(it->first).plane_coeffs = it->second.plane_coeffs;
    }
    return rec_planes;
}

std::string extractSceneName(std::string path) {
    size_t last_of;
    last_of = path.find_last_of("/");

    if (last_of == path.size()-1) {
        path.erase(last_of,1);
        last_of = path.find_last_of("/");
    }
    return path.substr(last_of+1, path.size()-1);
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
    std::string ppf_config_path_path="";
    pcl::console::parse(argc, argv, "-r", base_result_path);
    pcl::console::parse(argc, argv, "-c", ppf_config_path_path);

    //extract all scene folders
    if (!boost::filesystem::exists(room_path) || !boost::filesystem::is_directory(room_path)) {
        std::cout << room_path << " does not exist or is not a directory" << std::endl;
        return -1;
    }
    std::vector<std::string> all_scene_paths;
    for (boost::filesystem::directory_entry& scene : boost::filesystem::directory_iterator(room_path)) {
        if (boost::filesystem::is_directory(scene)) {
            std::string p = scene.path().string();
            if (p.find("scene", p.length()-8) !=std::string::npos)
                all_scene_paths.push_back(p);
        }
    }
    std::sort(all_scene_paths.begin(), all_scene_paths.end());

    std::string timestamp = getCurrentTime();
    base_result_path =  base_result_path + "/" + timestamp + (do_LV_before_matching ? "_withLV":"" ) + "_filterUnwantedObjects_clusterMatchingDiff_fullPipeline/";
    //----------------------------setup result folder----------------------------------
    //start at 1 because element 0 is scene1 without objects
    for (size_t idx = 1; idx < all_scene_paths.size(); idx++)
    {
        for (int k = idx + 1; k < all_scene_paths.size(); k ++)
        {
            std::string reference_path = all_scene_paths[idx];
            std::string current_path = all_scene_paths[k];
            //extract the two scene names
            std::string ref_scene_name = extractSceneName(reference_path);
            std::string curr_scene_name = extractSceneName(current_path);

            result_path = base_result_path + ref_scene_name + "-" + curr_scene_name + "/";
            boost::filesystem::create_directories(result_path);

            boost::filesystem::copy(ppf_config_path_path, result_path+"/config.ini");

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

            /// Input: Two reconstructed POI in map frame with RGB, Normals and Lables (?), coefficients of the plane/plane points
            /// Parameters:
            ///     (- downsample input --> no parameter, we do that anyway!)
            ///     - perform LV, if yes with which parameters
            ///     (- perform region growing, if yes which parameters --> we do that anyway)
            ///     - filter objects (based on size, planarity, color...)

            //-----------------------read convex hull points and transformations into map frame from input files-----------------------------------
            std::map<int, ReconstructedPlane> ref_rec_planes = prepareInputData(reference_path);
            std::map<int, ReconstructedPlane> curr_rec_planes = prepareInputData(current_path);

            for (std::map<int, ReconstructedPlane>::iterator ref_it = ref_rec_planes.begin(); ref_it != ref_rec_planes.end(); ref_it++ ) {
                pcl::io::savePCDFile(reference_path + "/convex_hull_" + std::to_string(ref_it->first) + ".pcd", *(ref_it->second.convex_hull_cloud));
            }
            for (std::map<int, ReconstructedPlane>::iterator curr_it = curr_rec_planes.begin(); curr_it != curr_rec_planes.end(); curr_it++ ) {
                pcl::io::savePCDFile(current_path + "/convex_hull_" + std::to_string(curr_it->first) + ".pcd", *(curr_it->second.convex_hull_cloud));
            }

            //iterate through all reference planes
            ChangeDetection change_detection(ppf_config_path_path);
            for (std::map<int, ReconstructedPlane>::iterator ref_it = ref_rec_planes.begin(); ref_it != ref_rec_planes.end(); ref_it++ ) {
                if (ref_it->second.cloud->empty()) {
                    ref_it->second.is_checked=true;
                    continue;
                }
                // find the closest plane in the current scene
                std::pair<int, ReconstructedPlane> closest_curr_element;
                float min_dist = std::numeric_limits<float>::max();
                for (std::map<int, ReconstructedPlane>::iterator curr_it = curr_rec_planes.begin(); curr_it != curr_rec_planes.end(); curr_it++ ) {
                    if (curr_it->second.cloud->empty()) {
                        curr_it->second.is_checked=true;
                        continue;
                    }
                    float dist = Point3D::squaredEuclideanDistance(ref_it->second.center_point, curr_it->second.center_point);
                    if (dist < min_dist) {
                        min_dist = dist;
                        closest_curr_element = *curr_it;
                    }
                }

                if (min_dist < 0.5 && !closest_curr_element.second.is_checked) {
                    ref_it->second.is_checked=true;
                    curr_rec_planes[closest_curr_element.first].is_checked = true;

                    std::cout << "-------------------------- " << ref_it->first << "-" << closest_curr_element.first << " --------------------------" << std::endl;
                    std::string plane_comparison_path = result_path + "/" + std::to_string(ref_it->first) + "_" + std::to_string(closest_curr_element.first);
                    boost::filesystem::create_directories(plane_comparison_path);

                    std::string merge_object_parts_folder = plane_comparison_path + "/mergeObjectParts";
                    boost::filesystem::create_directory(merge_object_parts_folder);

                    pcl::io::savePCDFile(plane_comparison_path + "/ref_cloud.pcd", *ref_it->second.cloud);
                    pcl::io::savePCDFile(plane_comparison_path + "/curr_cloud.pcd", *closest_curr_element.second.cloud);


                    std::vector<DetectedObject> ref_result, curr_result;
                    change_detection.init(ref_it->second.cloud, closest_curr_element.second.cloud,
                                          ref_it->second.plane_coeffs, closest_curr_element.second.plane_coeffs,
                                          ref_it->second.convex_hull_cloud, closest_curr_element.second.convex_hull_cloud,
                                          ppf_model_path, plane_comparison_path, merge_object_parts_folder);
                    change_detection.compute(ref_result, curr_result);

                    //TODO check if all existing model folders are also present in ref_result
                    //all detected objects labeled as removed (ref_objects) or new (curr_objects) could be placed on another plane
                    updateDetectedObjects(ref_result, curr_result);


//                    //after collecting potential new and removed objects from the plane, try to match them
//                    if (pot_removed_obj.size() != 0 && pot_new_obj.size() != 0) {
//                        //transform map into vec to be able to call object matching
//                        std::vector<DetectedObject> pot_rem_obj_vec, pot_new_obj_vec;
//                        pot_rem_obj_vec = fromMapToValVec(pot_removed_obj);
//                        pot_new_obj_vec = fromMapToValVec(pot_new_obj);
//                        ObjectMatching matching(pot_rem_obj_vec, pot_new_obj_vec, ppf_model_path, ppf_config_path_path);
//                        ref_result.clear(); curr_result.clear();
//                        matching.compute(ref_result, curr_result);

//                        ChangeDetection::mergeObjectParts(ref_result, merge_object_parts_folder);
//                        ChangeDetection::mergeObjectParts(curr_result, merge_object_parts_folder);

//                        updateDetectedObjects(ref_result, curr_result);
//                    }
                }
            }


            //extract objects from all planes where is_checked=false and try to match them
            for (std::map<int, ReconstructedPlane>::iterator ref_it = ref_rec_planes.begin(); ref_it != ref_rec_planes.end(); ref_it++ ) {
                if (ref_it->second.is_checked == false) {
                    std::string plane_comparison_path = result_path + "/ref_" + std::to_string(ref_it->first);
                    boost::filesystem::create_directories(plane_comparison_path);
                    pcl::io::savePCDFile(plane_comparison_path + "/ref_cloud.pcd", *ref_it->second.cloud);

                    std::string merge_object_parts_folder = plane_comparison_path + "/mergeObjectParts";
                    boost::filesystem::create_directory(merge_object_parts_folder);

                    std::vector<DetectedObject> ref_result, curr_result;
                    pcl::PointCloud<PointNormal>::Ptr fake_cloud(new pcl::PointCloud<PointNormal>);
                    pcl::PointCloud<pcl::PointXYZ>::Ptr fake_hull_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                    change_detection.init(ref_it->second.cloud, fake_cloud,
                                          ref_it->second.plane_coeffs, Vector4f_NotAligned(),
                                          ref_it->second.convex_hull_cloud, fake_hull_cloud,
                                          ppf_model_path, plane_comparison_path, merge_object_parts_folder);
                    change_detection.compute(ref_result, curr_result);
                    updateDetectedObjects(ref_result, curr_result);
                }
            }
            for (std::map<int, ReconstructedPlane>::iterator curr_it = curr_rec_planes.begin(); curr_it != curr_rec_planes.end(); curr_it++ ) {
                if (curr_it->second.is_checked == false) {
                    std::string plane_comparison_path = result_path + "/curr_" + std::to_string(curr_it->first);
                    boost::filesystem::create_directories(plane_comparison_path);
                    pcl::io::savePCDFile(plane_comparison_path + "/curr_cloud.pcd", *curr_it->second.cloud);

                    std::string merge_object_parts_folder = plane_comparison_path + "/mergeObjectParts";
                    boost::filesystem::create_directory(merge_object_parts_folder);

                    std::vector<DetectedObject> ref_result, curr_result;
                    pcl::PointCloud<PointNormal>::Ptr fake_cloud(new pcl::PointCloud<PointNormal>);
                    pcl::PointCloud<pcl::PointXYZ>::Ptr fake_hull_cloud(new pcl::PointCloud<pcl::PointXYZ>);
                    change_detection.init(fake_cloud, curr_it->second.cloud,
                                          Vector4f_NotAligned(), curr_it->second.plane_coeffs,
                                          fake_hull_cloud, curr_it->second.convex_hull_cloud,
                                          ppf_model_path, plane_comparison_path, merge_object_parts_folder);
                    change_detection.compute(ref_result, curr_result);
                    updateDetectedObjects(ref_result, curr_result);
                }
            }

            //one last chance to match objects
            //after collecting potential new and removed objects from the plane, try to match them
            if (pot_removed_obj.size() != 0 && pot_new_obj.size() != 0) {
                std::string merge_object_parts_folder = result_path + "/leftover_mergeObjectParts";
                boost::filesystem::create_directory(merge_object_parts_folder);

                //transform map into vec to be able to call object matching
                std::vector<DetectedObject> pot_rem_obj_vec, pot_new_obj_vec;
                pot_rem_obj_vec = fromMapToValVec(pot_removed_obj);
                pot_new_obj_vec = fromMapToValVec(pot_new_obj);
                ObjectMatching matching(pot_rem_obj_vec, pot_new_obj_vec, ppf_model_path, ppf_config_path_path);
                std::vector<DetectedObject> ref_result, curr_result;
                matching.compute(ref_result, curr_result);

                ChangeDetection::mergeObjectParts(ref_result, merge_object_parts_folder);
                ChangeDetection::mergeObjectParts(curr_result, merge_object_parts_folder);

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



            //put all planes together in one file as reference
            pcl::PointCloud<PointNormal>::Ptr ref_cloud_merged(new pcl::PointCloud<PointNormal>);
            pcl::PointCloud<PointNormal>::Ptr curr_cloud_merged(new pcl::PointCloud<PointNormal>);

            for (std::map<int, ReconstructedPlane>::iterator ref_it = ref_rec_planes.begin(); ref_it != ref_rec_planes.end(); ref_it++ ) {
                //crop cloud according to the convex hull points, find min and max values in x and y direction
                pcl::PointXYZ max_hull_pt, min_hull_pt;
                pcl::getMinMax3D(*(ref_it->second.convex_hull_cloud), min_hull_pt, max_hull_pt);

                //add some alignment because hull points were computed from a full room reconstruction that may include drift
                pcl::PointCloud<PointNormal>::Ptr cropped_cloud(new pcl::PointCloud<PointNormal>);
                pcl::PassThrough<PointNormal> pass;
                pass.setInputCloud(ref_it->second.cloud);
                pass.setFilterFieldName("x");
                pass.setFilterLimits(min_hull_pt.x - 0.15, max_hull_pt.x + 0.15);
                pass.setKeepOrganized(true);
                pass.filter(*cropped_cloud);
                pass.setInputCloud(cropped_cloud);
                pass.setFilterFieldName("y");
                pass.setFilterLimits(min_hull_pt.y - 0.15, max_hull_pt.y + 0.15);
                pass.setKeepOrganized(true);
                pass.filter(*cropped_cloud);

                *ref_cloud_merged += *cropped_cloud;
            }
            pcl::PointCloud<PointNormal>::Ptr ref_merged_ds(new pcl::PointCloud<PointNormal>);
            ref_merged_ds = downsampleCloudVG(ref_cloud_merged, 0.01);
            pcl::io::savePCDFile(result_path + "/ref_cloud_merged.pcd", *ref_merged_ds);

            for (std::map<int, ReconstructedPlane>::iterator curr_it = curr_rec_planes.begin(); curr_it != curr_rec_planes.end(); curr_it++ ) {
                //crop cloud according to the convex hull points, find min and max values in x and y direction
                pcl::PointXYZ max_hull_pt, min_hull_pt;
                pcl::getMinMax3D(*(curr_it->second.convex_hull_cloud), min_hull_pt, max_hull_pt);

                //add some alignment because hull points were computed from a full room reconstruction that may include drift
                pcl::PointCloud<PointNormal>::Ptr cropped_cloud(new pcl::PointCloud<PointNormal>);
                pcl::PassThrough<PointNormal> pass;
                pass.setInputCloud(curr_it->second.cloud);
                pass.setFilterFieldName("x");
                pass.setFilterLimits(min_hull_pt.x - 0.15, max_hull_pt.x + 0.15);
                pass.setKeepOrganized(true);
                pass.filter(*cropped_cloud);
                pass.setInputCloud(cropped_cloud);
                pass.setFilterFieldName("y");
                pass.setFilterLimits(min_hull_pt.y - 0.15, max_hull_pt.y + 0.15);
                pass.setKeepOrganized(true);
                pass.filter(*cropped_cloud);

                *curr_cloud_merged += *cropped_cloud;
            }
            pcl::PointCloud<PointNormal>::Ptr curr_merged_ds(new pcl::PointCloud<PointNormal>);
            curr_merged_ds = downsampleCloudVG(curr_cloud_merged, 0.01);
            pcl::io::savePCDFile(result_path + "/curr_cloud_merged.pcd", *curr_merged_ds);


            //visualization with PCLViewer
            //copy the fused cloud and add colored points from detected objects (e.g. removed ones red, new ones green, and displaced ones r and g random and b high number)
            //ObjectVisualization vis(ref_cloud_merged, curr_cloud_merged, removed_obj_vec, new_obj_vec,
            //                        ref_dis_obj_vec, curr_dis_obj_vec, ref_static_obj_vec, curr_static_obj_vec);
            //vis.visualize();

        }
    }
}


