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

std::vector<DetectedObject> new_obj;
std::vector<DetectedObject> removed_obj;
std::vector<DetectedObject> pot_new_obj;
std::vector<DetectedObject> pot_removed_obj;
std::vector<DetectedObject> ref_displaced_obj;
std::vector<DetectedObject> curr_displaced_obj;

template <typename DetectedObject>
ostream& operator<<(ostream& output, std::vector<DetectedObject> const& values)
{
    for (auto const & value : values)
    {
        const int &id =  value.getID();
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


int main(int argc, char* argv[])
{
    /// Check arguments and print info
    if (argc < 4) {
        pcl::console::print_info("\n\
                                 -- Object change detection based on reconstructions of a plane of interest at two different timestamps -- : \n\
                                 \n\
                                 Syntax: %s reference_plane  current_plane \n\
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

    /// Input: Two reconstructed POI in map frame with RGB, Normals and Lables (?), coefficients of the plane/plane points
    /// Parameters:
    ///     (- downsample input --> no parameter, we do that anyway!)
    ///     - perform LV, if yes with which parameters
    ///     (- perform region growing, if yes which parameters --> we do that anyway)
    ///     - filter objects (based on size, planarity, color...)

    //-----------------------read scene input files-----------------------------------
    if(!boost::filesystem::exists(reference_path)) {
        std::cerr << reference_path << " does not exist!" << std::endl;
        return (-1);
    }
    if(!boost::filesystem::exists(current_path)) {
        std::cerr << current_path << " does not exist!" << std::endl;
        return (-1);
    }

    pcl::PointCloud<PointNormal>::Ptr ref_cloud (new pcl::PointCloud<PointNormal>);
    if (readInput(reference_path, ref_cloud))
        std::cout << "Loaded reference plane successfully with " << ref_cloud->size() << " points" << std::endl;

    pcl::PointCloud<PointNormal>::Ptr curr_cloud (new pcl::PointCloud<PointNormal>);
    if (readInput(current_path, curr_cloud))
        std::cout << "Loaded current plane successfully with " << curr_cloud->size() << " points" << std::endl;

    pcl::io::savePCDFile(result_path + "/ref_cloud.pcd", *ref_cloud);
    pcl::io::savePCDFile(result_path + "/curr_cloud.pcd", *curr_cloud);

    //TODO Read plane coeffs and convex hull points from DB
    Eigen::Vector4f curr_plane_coeffs(-0.00958200544119, 0.00314281438477, 0.999949157238, -0.41155308485);
    Eigen::Vector4f ref_plane_coeffs(-0.00958200544119, 0.00314281438477, 0.999949157238, -0.41155308485);
    //std::vector<pcl::PointXYZ> curr_convex_hull_pts;
    //std::vector<pcl::PointXYZ> ref_convex_hull_pts;
    pcl::PointCloud<pcl::PointXYZ>::Ptr curr_hull_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr ref_hull_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile("/home/edith/test_data_MarkusLeitner/table_test_data/table_hull.pcd", *curr_hull_cloud);
    pcl::copyPointCloud(*curr_hull_cloud, *ref_hull_cloud);

    std::vector<DetectedObject> ref_result, curr_result;
    ChangeDetection change_detection(ppf_config_path_path);
    change_detection.init(ref_cloud, curr_cloud, ref_plane_coeffs, curr_plane_coeffs, curr_hull_cloud, ref_hull_cloud, result_path);
    change_detection.compute(ref_result, curr_result);

    //all detected objects labeled as removed (ref_objects) or new (curr_objects) could be placed on another plane
    for (DetectedObject ro : ref_result) {
        if (ro.state_ == ObjectState::REMOVED) {
            pot_removed_obj.push_back(ro);
            //means that there was a partial match and a new ppf-model is needed
            if (ro.object_folder_path_ == "") {
                //there should already exist a folder
                std::string orig_path = result_path + "/model_objects/" + std::to_string(ro.getID());
                if (boost::filesystem::exists(orig_path)) {
                    boost::filesystem::path dest_folder = result_path + "/model_objects_matched/" + std::to_string(ro.getID());
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
        } else if (ro.state_ == ObjectState::DISPLACED) {
            ref_displaced_obj.push_back(ro);
        } //the state STATIC can be ignored
    }
    for (DetectedObject co : curr_result) {
        if (co.state_ == ObjectState::NEW) {
            pot_new_obj.push_back(co);
        } else if (co.state_ == ObjectState::DISPLACED) {
            curr_displaced_obj.push_back(co);
        } //the state STATIC can be ignored
    }

    //TODO do that for several planes

    //after collecting potential new and removed objects from all planes, try to match them
    if (pot_removed_obj.size() != 0 && pot_new_obj.size() != 0) {
        ObjectMatching matching(pot_removed_obj, pot_new_obj, result_path + "/model_objects/", ppf_config_path_path);
        ref_result.clear(); curr_result.clear();
        matching.compute(ref_result, curr_result);
        for (DetectedObject ro : ref_result) {
            if (ro.state_ == ObjectState::REMOVED) {
                removed_obj.push_back(ro);
            } else if (ro.state_ == ObjectState::DISPLACED) {
                ref_displaced_obj.push_back(ro);
            } //the state STATIC can be ignored
        }
        for (DetectedObject co : curr_result) {
            if (co.state_ == ObjectState::NEW) {
                new_obj.push_back(co);
            } else if (co.state_ == ObjectState::DISPLACED) {
                curr_displaced_obj.push_back(co);
            } //the state STATIC can be ignored
        }
    } else {
        //one of them is empty
        removed_obj.insert(removed_obj.end(), pot_removed_obj.begin(), pot_removed_obj.end());
        new_obj.insert(new_obj.end(), pot_new_obj.begin(), pot_new_obj.end());
    }

    std::cout << "FINAL RESULT" << std::endl;
    std::cout << "Removed objects: " << removed_obj << std::endl;
    std::cout << "New objects: " << new_obj << std::endl;
    std::cout << "Displaced objects in reference: " << ref_displaced_obj << std::endl;
    std::cout << "Displaced objects in current: " << curr_displaced_obj << std::endl;

    //TODO put all plane reconstructions into one cloud
    //visualization
    //put all planes together in one pcd-file as reference
    //copy the fused cloud and add colored points from detected objects (e.g. removed ones red, new ones green, and displaced ones r and g random and b high number)
    ObjectVisualization vis(ref_cloud, curr_cloud, removed_obj, new_obj, ref_displaced_obj, curr_displaced_obj);
    vis.visualize();

}


