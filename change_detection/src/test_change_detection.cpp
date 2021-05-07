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

void updateModelInput(DetectedObject ro, std::string ppf_model_path, std::string result_path) {
    //there should already exist a folder
    std::string orig_path = ppf_model_path + "/" + std::to_string(ro.getID());
    if (boost::filesystem::exists(orig_path)) {
        boost::filesystem::path dest_folder = result_path + "/model_partially_matched/" + std::to_string(ro.getID());
        boost::filesystem::create_directories(dest_folder);
        //copy the folder to another directory. The model is already matched and should not be used anymore
        for (const auto& dirEnt : boost::filesystem::recursive_directory_iterator{orig_path})
        {
            const auto& path = dirEnt.path();
            boost::filesystem::copy(path, dest_folder / path.filename());
            std::cout << "Copy " << path << " to " << (dest_folder / path.filename()) << std::endl;
        }
    }
    boost::filesystem::remove_all(orig_path);
    boost::filesystem::create_directories(orig_path);
    pcl::io::savePCDFile(orig_path + "/3D_model.pcd", *ro.object_cloud_); //PPF will create a new model with the new cloud
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

            transformation_map[table_nr].insert(std::make_pair(occurrance, mat.inverse())); //from camera frame to map frame
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
    //we assume that the the ply-files are numbered continuously
    const std::regex ply_filter( "table_[0-9]+.ply" );
    for(auto & plane_nr_path : boost::filesystem::directory_iterator(data_path + "/planes"))
    {
        if (boost::filesystem::is_directory(plane_nr_path.status()))
        {
            int count = 0;
            int plane_nr = std::stoi(plane_nr_path.path().filename().string());
            for(auto & occ_nr_path : boost::filesystem::directory_iterator(plane_nr_path))
            {
                if (!boost::filesystem::is_directory(occ_nr_path.status()))
                {
                    if(std::regex_match(occ_nr_path.path().filename().string(), ply_filter ) )
                        count++;
                }
            }

            pcl::PointCloud<PointNormal>::Ptr plane_cloud(new pcl::PointCloud<PointNormal>);
            if (count == 1) {
                readInput(plane_nr_path.path().string() + "/table_0.ply", plane_cloud);
                //transform the ply
                Matrix4f_NotAligned & mat = transformations[plane_nr][0];
                pcl::transformPointCloudWithNormals(*plane_cloud, *plane_cloud, mat);
            } else if (count > 1){ //merge the plys
                readInput(plane_nr_path.path().string() + "/table_0.ply", plane_cloud);

                //transform the cloud because hull points are in map frame and not camera frame
                Matrix4f_NotAligned & mat = transformations[plane_nr][0];
                pcl::transformPointCloudWithNormals(*plane_cloud, *plane_cloud, mat);

                //crop cloud according to the convex hull points, find min and max values in x and y direction
                pcl::PointXYZ max_hull_pt, min_hull_pt;
                pcl::getMinMax3D(*(center_and_convexHull[plane_nr].convex_hull_cloud), min_hull_pt, max_hull_pt);

                pcl::PointCloud<PointNormal>::Ptr cropped_target_cloud(new pcl::PointCloud<PointNormal>);
                pcl::PassThrough<PointNormal> pass;
                pass.setInputCloud(plane_cloud);
                pass.setFilterFieldName("x");
                pass.setFilterLimits(min_hull_pt.x - 0.1, max_hull_pt.x + 0.1);
                pass.setKeepOrganized(false);
                pass.filter(*cropped_target_cloud);
                pass.setInputCloud(cropped_target_cloud);
                pass.setFilterFieldName("y");
                pass.setFilterLimits(min_hull_pt.y - 0.1, max_hull_pt.y + 0.1);
                pass.filter(*cropped_target_cloud);
                pcl::io::savePCDFileBinary(plane_nr_path.path().string() + "/cropped_target_cloud.pcd", *cropped_target_cloud);
                for (size_t i = 1; i < count; i++ ) {
                    pcl::PointCloud<PointNormal>::Ptr source_cloud(new pcl::PointCloud<PointNormal>);
                    pcl::PointCloud<PointNormal>::Ptr cropped_source_cloud(new pcl::PointCloud<PointNormal>);
                    readInput(plane_nr_path.path().string() + "/table_"+ std::to_string(i) +".ply", source_cloud);

                    //transform the cloud because hull points are in map frame and not camera frame
                    Matrix4f_NotAligned & mat = transformations[plane_nr][i];
                    pcl::transformPointCloudWithNormals(*source_cloud, *source_cloud, mat);

                    pass.setInputCloud(source_cloud);
                    pass.setFilterFieldName("x");
                    pass.setFilterLimits(min_hull_pt.x - 0.1, max_hull_pt.x + 0.1);
                    pass.filter(*cropped_source_cloud);
                    pass.setInputCloud(cropped_source_cloud);
                    pass.setFilterFieldName("y");
                    pass.setFilterLimits(min_hull_pt.y - 0.1, max_hull_pt.y + 0.1);
                    pass.filter(*cropped_source_cloud);

                    pcl::io::savePCDFileBinary(plane_nr_path.path().string() + "/cropped_source_cloud.pcd", *cropped_source_cloud);

                    //align to table_1.ply
                    pcl::PointCloud<PointNormal>::Ptr plane_registered(new pcl::PointCloud<PointNormal>());
                    pcl::IterativeClosestPointWithNormals<PointNormal, PointNormal> icp;
                    icp.setInputSource(cropped_source_cloud);
                    icp.setInputTarget(cropped_target_cloud);
                    icp.setMaxCorrespondenceDistance(0.1);
                    icp.setRANSACOutlierRejectionThreshold(0.009);
                    icp.setMaximumIterations(500);
                    icp.setTransformationEpsilon (1e-9);
                    icp.setTransformationRotationEpsilon(1 - 1e-15); //epsilon is the cos(angle)
                    icp.align(*plane_registered);
                    std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
                    *cropped_target_cloud += *plane_registered;
                }
                pcl::io::savePCDFileBinary(plane_nr_path.path().string() + "/merged_plane_clouds.pcd", *plane_cloud);
            }
            //save in vector of planeStructs
            ReconstructedPlane rec_plane;
            rec_plane.cloud = plane_cloud;
            rec_plane.is_checked = false;
            rec_planes[plane_nr] = rec_plane;
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

int main(int argc, char* argv[])
{
    /// Check arguments and print info
    if (argc < 4) {
        pcl::console::print_info("\n\
                                 -- Object change detection based on reconstructions of a plane of interest at two different timestamps -- : \n\
                                 \n\
                                 Syntax: %s reference_path  current_path \n\
                                 [Options] \n\
                                 -r path, where results should be stored, a folder with date and time gets created there \n\
                                 -c config path for ppf params",
                                 argv[0]);
        return(1);
    }

    /// Parse command line arguments
    std::string reference_path = argv[1];
    std::string current_path = argv[2];
    std::string result_path="";
    std::string ppf_config_path_path="";
    pcl::console::parse(argc, argv, "-r", result_path);
    pcl::console::parse(argc, argv, "-c", ppf_config_path_path);

    //----------------------------setup result folder----------------------------------
    std::string timestamp = getCurrentTime();
    result_path =  result_path + "/" + timestamp;
    boost::filesystem::create_directories(result_path);

    //----------------------------setup ppf model folder-------------------------------
    std::string ppf_model_path = result_path + "/model_objects/";
    boost::filesystem::create_directories(ppf_model_path);

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
    for (std::map<int, ReconstructedPlane>::iterator ref_it = ref_rec_planes.begin(); ref_it != ref_rec_planes.end(); ref_it++ ) {
        // find the closest plane in the current scene
        std::pair<int, ReconstructedPlane> closest_curr_element;
        float min_dist = std::numeric_limits<float>::max();
        for (std::map<int, ReconstructedPlane>::iterator curr_it = curr_rec_planes.begin(); curr_it != curr_rec_planes.end(); curr_it++ ) {
            float dist = Point3D::squaredEuclideanDistance(ref_it->second.center_point, curr_it->second.center_point);
            if (dist < min_dist) {
                min_dist = dist;
                closest_curr_element = *curr_it;
            }
        }

        if (min_dist < 0.5 && !closest_curr_element.second.is_checked) {
            ref_it->second.is_checked=true;
            curr_rec_planes[closest_curr_element.first].is_checked = true;

            std::string plane_comparison_path = result_path + "/" + std::to_string(ref_it->first) + "_" + std::to_string(closest_curr_element.first);
            boost::filesystem::create_directories(plane_comparison_path);

            pcl::io::savePCDFile(plane_comparison_path + "/ref_cloud.pcd", *ref_it->second.cloud);
            pcl::io::savePCDFile(plane_comparison_path + "/curr_cloud.pcd", *closest_curr_element.second.cloud);


            std::vector<DetectedObject> ref_result, curr_result;
            ChangeDetection change_detection(ppf_config_path_path);
            change_detection.init(ref_it->second.cloud, closest_curr_element.second.cloud,
                                  ref_it->second.plane_coeffs, closest_curr_element.second.plane_coeffs,
                                  ref_it->second.convex_hull_cloud, closest_curr_element.second.convex_hull_cloud,
                                  ppf_model_path, plane_comparison_path);
            change_detection.compute(ref_result, curr_result);

            //TODO check if all existing model folders are also present in ref_result
            //all detected objects labeled as removed (ref_objects) or new (curr_objects) could be placed on another plane
            for (DetectedObject ro : ref_result) {
                if (ro.state_ == ObjectState::REMOVED) {
                    pot_removed_obj[ro.getID()] = ro;
                    //means that there was a partial match and a new ppf-model is needed
                    if (ro.object_folder_path_ == "") {
                        updateModelInput(ro, ppf_model_path, result_path);
                    }
                } else if (ro.state_ == ObjectState::DISPLACED) {
                    ref_displaced_obj[ro.getID()] = ro;
                } //the state STATIC can be ignored
            }
            for (DetectedObject co : curr_result) {
                if (co.state_ == ObjectState::NEW) {
                    pot_new_obj[co.getID()] = co;
                } else if (co.state_ == ObjectState::DISPLACED) {
                    curr_displaced_obj[co.getID()] = co;
                } //the state STATIC can be ignored
            }


            //after collecting potential new and removed objects from all planes, try to match them
            if (pot_removed_obj.size() != 0 && pot_new_obj.size() != 0) {
                //transform map into vec to be able to call object matching
                std::vector<DetectedObject> pot_rem_obj_vec, pot_new_obj_vec;
                pot_rem_obj_vec = fromMapToValVec(pot_removed_obj);
                pot_new_obj_vec = fromMapToValVec(pot_new_obj);
                ObjectMatching matching(pot_rem_obj_vec, pot_new_obj_vec, ppf_model_path, ppf_config_path_path);
                ref_result.clear(); curr_result.clear();
                matching.compute(ref_result, curr_result);
                for (DetectedObject ro : ref_result) {
                    if (ro.state_ == ObjectState::REMOVED) {
                        pot_removed_obj[ro.getID()] = ro;
                        //means that there was a partial match and a new ppf-model is needed
                        if (ro.object_folder_path_ == "") {
                            updateModelInput(ro, ppf_model_path, result_path);
                        }
                    } else if (ro.state_ == ObjectState::DISPLACED) {
                        ref_displaced_obj[ro.getID()] = ro;
                    } //the state STATIC can be ignored
                }
                for (DetectedObject co : curr_result) {
                    if (co.state_ == ObjectState::NEW) {
                        pot_new_obj[co.getID()] = co;
                    } else if (co.state_ == ObjectState::DISPLACED) {
                        curr_displaced_obj[co.getID()] = co;
                    } //the state STATIC can be ignored
                }
            }
        }
    }


    //TODO extract objects from all planes where is_checked=false and try to match them



    //all pot. moved objects are in the end either removed or new
    removed_obj = pot_removed_obj;
    new_obj = pot_new_obj;

    std::cout << "FINAL RESULT" << std::endl;
    std::cout << "Removed objects: " << removed_obj << std::endl;
    std::cout << "New objects: " << new_obj << std::endl;
    std::cout << "Displaced objects in reference: " << ref_displaced_obj << std::endl;
    std::cout << "Displaced objects in current: " << curr_displaced_obj << std::endl;


    //create point clouds of detected objects to save results as pcd-files
    pcl::PointCloud<PointNormal>::Ptr ref_removed_objects_cloud(new pcl::PointCloud<PointNormal>);
    pcl::PointCloud<PointNormal>::Ptr ref_displaced_objects_cloud(new pcl::PointCloud<PointNormal>);
    pcl::PointCloud<PointNormal>::Ptr curr_new_objects_cloud(new pcl::PointCloud<PointNormal>);
    pcl::PointCloud<PointNormal>::Ptr curr_displaced_objects_cloud(new pcl::PointCloud<PointNormal>);

    for (auto const & o : removed_obj) {
        *ref_removed_objects_cloud += *(o.second.object_cloud_);
    }
    if (!ref_removed_objects_cloud->empty())
        pcl::io::savePCDFile(result_path + "/ref_removed_objects.pcd", *ref_removed_objects_cloud);

    for (auto const & o : ref_displaced_obj) {
        *ref_displaced_objects_cloud += *(o.second.object_cloud_);
    }
    if (!ref_displaced_objects_cloud->empty())
        pcl::io::savePCDFile(result_path + "/ref_displaced_objects.pcd", *ref_displaced_objects_cloud);

    for (auto const & o : new_obj) {
        *curr_new_objects_cloud += *(o.second.object_cloud_);
    }
    if (!curr_new_objects_cloud->empty())
        pcl::io::savePCDFile(result_path + "/curr_new_objects.pcd", *curr_new_objects_cloud);

    for (auto const & o : curr_displaced_obj) {
        *curr_displaced_objects_cloud += *(o.second.object_cloud_);
    }
    if (!curr_displaced_objects_cloud->empty())
        pcl::io::savePCDFile(result_path + "/curr_displaced_objects.pcd", *curr_displaced_objects_cloud);


    //visualization
    //put all planes together in one file as reference
    //copy the fused cloud and add colored points from detected objects (e.g. removed ones red, new ones green, and displaced ones r and g random and b high number)
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
    ref_merged_ds = ChangeDetection::downsampleCloud(ref_cloud_merged, 0.01);
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
    curr_merged_ds = ChangeDetection::downsampleCloud(curr_cloud_merged, 0.01);
    pcl::io::savePCDFile(result_path + "/curr_cloud_merged.pcd", *curr_merged_ds);


    std::vector<DetectedObject> removed_obj_vec, new_obj_vec, ref_dis_obj_vec, curr_dis_obj_vec;
    removed_obj_vec = fromMapToValVec(removed_obj);
    new_obj_vec = fromMapToValVec(new_obj);
    ref_dis_obj_vec = fromMapToValVec(ref_displaced_obj);
    curr_dis_obj_vec = fromMapToValVec(curr_displaced_obj);

    ObjectVisualization vis(ref_cloud_merged, curr_cloud_merged, removed_obj_vec, new_obj_vec, ref_dis_obj_vec, curr_dis_obj_vec);
    vis.visualize();


}


