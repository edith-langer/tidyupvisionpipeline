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
    if (argc < 3) {
        pcl::console::print_info("\n\
                                 -- Object change detection based on reconstructions of a plane of interest at two different timestamps -- : \n\
                                 \n\
                                 Syntax: %s reference_plane  current_plane \n\
                                 [Options] \n\
                                 -r path, where results should be stored, a folder with date and time gets created there ",
                                 argv[0]);
        return(1);
    }

    /// Parse command line arguments
    std::string reference_path = argv[1];
    std::string current_path = argv[2];
    std::string result_path="";
    pcl::console::parse(argc, argv, "-r", result_path);

    //----------------------------setup result folder----------------------------------
    std::string timestamp = getCurrentTime();
    result_path =  result_path + timestamp;
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

    //Read plane coeffs and convex hull points from DB
    Eigen::Vector4f curr_plane_coeffs;
    Eigen::Vector4f ref_plane_coeffs;
    std::vector<pcl::PointXYZ> curr_convex_hull_pts;
    std::vector<pcl::PointXYZ> ref_convex_hull_pts;


    ChangeDetectionResult ref_result, curr_result;
    ChangeDetection change_detection(ref_cloud, curr_cloud, ref_plane_coeffs, curr_plane_coeffs, ref_convex_hull_pts, curr_convex_hull_pts, result_path);
    change_detection.compute(ref_result, curr_result);
}


