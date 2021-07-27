#include <boost/filesystem.hpp>

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <regex>

#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/filter.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/registration/icp.h>
#include <pcl/common/common.h>

typedef pcl::PointXYZRGBNormal PointNormal;
typedef pcl::PointXYZRGB PointRGB;
typedef pcl::PointXYZRGBL PointLabel;

typedef Eigen::Matrix<float,4,4,Eigen::DontAlign> Matrix4f_NotAligned;

using namespace  std;

std::vector<std::string> merged_paths;
std::vector<std::string> icp_not_converged;

template <typename T>
ostream& operator<<(ostream& os, const vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i != v.size() - 1)
            os << "\n ";
    }
    os << "]\n";
    return os;
}

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

void prepareInputData(std::string data_path) {
    std::map<int, ReconstructedPlane> rec_planes;
    if(!boost::filesystem::exists(data_path) && !boost::filesystem::is_directory(data_path) ) {
        std::cerr << "The data path does not exist or is not a directory " << data_path << std::endl;
        return;
    }

    //TODO Read plane coeffs and convex hull points from DB
    std::map<int, ReconstructedPlane> center_and_convexHull= convexHullPtsParser(data_path + "/table.txt");
    std::map<int, std::map<int, Matrix4f_NotAligned>> transformations = transformationParser(data_path + "/log.txt");

    //iterate through all plane folders and extract the reconstructed ply-files
    //we assume that the the ply-files are numbered continuously
    std::string post_str = "_fusedPoses_i100_biggerLambda";
    const std::regex ply_filter( "table_[0-9]+" + post_str + ".ply" );
    int plane_nr = 0;
    bool found_plane_folder = true;
    while (found_plane_folder) {
        std::string plane_nr_path = data_path + "/planes/" + std::to_string(plane_nr);
        if (boost::filesystem::exists(plane_nr_path) && boost::filesystem::is_directory(plane_nr_path)) {
            int count = 0;
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
                readInput(plane_nr_path + "/table_0" + post_str + ".ply", plane_cloud);
                //transform the ply
                Matrix4f_NotAligned & mat = transformations[plane_nr][0];
                pcl::transformPointCloudWithNormals(*plane_cloud, *plane_cloud, mat);
                pcl::io::savePCDFileBinary(plane_nr_path + "/merged_plane_clouds_" + post_str + ".pcd", *plane_cloud);
                ReconstructedPlane c_hull = center_and_convexHull.at(plane_nr);
                pcl::io::savePCDFileBinary(plane_nr_path + "/convex_hull.pcd", *c_hull.convex_hull_cloud);
            } else if (count > 1){ //merge the plys
                readInput(plane_nr_path + "/table_0" + post_str + ".ply", plane_cloud);

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
                if (!cropped_target_cloud->empty())
                    pcl::io::savePCDFileBinary(plane_nr_path + "/cropped_target_cloud_" + post_str + ".pcd", *cropped_target_cloud);
                for (size_t i = 1; i < count; i++ ) {
                    std::cout << "Plane " << plane_nr << "; table " << i << std::endl;
                    pcl::PointCloud<PointNormal>::Ptr source_cloud(new pcl::PointCloud<PointNormal>);
                    pcl::PointCloud<PointNormal>::Ptr cropped_source_cloud(new pcl::PointCloud<PointNormal>);
                    readInput(plane_nr_path + "/table_"+ std::to_string(i) + post_str + ".ply", source_cloud);

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

                    //this can happen if the table extractor method does not work perfectly
                    if (cropped_source_cloud->empty())
                        continue;

                    pcl::io::savePCDFileBinary(plane_nr_path + "/cropped_source_cloud_" + post_str +"_" + std::to_string(i) + ".pcd", *cropped_source_cloud);

                    if (cropped_target_cloud->empty())
                        cropped_target_cloud = cropped_source_cloud;

                    //align to table_1.ply
                    pcl::PointCloud<PointNormal>::Ptr plane_registered(new pcl::PointCloud<PointNormal>());
                    pcl::IterativeClosestPointWithNormals<PointNormal, PointNormal> icp;
                    //pcl::IterativeClosestPoint<PointNormal, PointNormal> icp;
                    icp.setInputSource(cropped_source_cloud);
                    icp.setInputTarget(cropped_target_cloud);
                    icp.setMaxCorrespondenceDistance(0.1);
                    icp.setRANSACOutlierRejectionThreshold(0.009);
                    icp.setMaximumIterations(500);
                    icp.setTransformationEpsilon (1e-9);
                    icp.setTransformationRotationEpsilon(1 - 1e-15); //epsilon is the cos(angle)
                    icp.align(*plane_registered);
                    std::cout << "has converged:" << icp.hasConverged() << " score: " << icp.getFitnessScore() << std::endl;
                    if (icp.hasConverged())
                        *cropped_target_cloud += *plane_registered;
                    else
                        icp_not_converged.push_back(plane_nr_path);
                }
                plane_cloud = cropped_target_cloud;
                if (!plane_cloud->empty())
                    merged_paths.push_back(plane_nr_path);
                    pcl::io::savePCDFileBinary(plane_nr_path + "/merged_plane_clouds_" + post_str + ".pcd", *plane_cloud);
            }
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
}

int main(int argc, char* argv[])
{
    std::string data_path = argv[1];

    const std::regex scene_filter("scene[2-9]+");
    for(auto & scene_path : boost::filesystem::directory_iterator(data_path))
    {
        if (boost::filesystem::is_directory(scene_path.status()))
        {
            if(std::regex_match(scene_path.path().filename().string(), scene_filter ) ) {
                std::cout << scene_path.path().string() << std::endl;
                prepareInputData(scene_path.path().string());
            }
        }
    }
    std::cout << merged_paths<< std::endl;
    std::cout << "ICP did not converge for " << std::endl;
    std::cout << icp_not_converged << std::endl;
}


