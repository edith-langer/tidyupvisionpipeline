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
#include <fstream>
#include <stdlib.h>
#include <chrono>

#include <pcl/search/search.h>
#include <pcl/octree/octree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>


typedef pcl::PointXYZRGBNormal PointNormal;
typedef pcl::PointXYZRGB PointRGB;
typedef pcl::PointXYZRGBL PointLabel;

boost::filesystem::path ds_file_name = "merged_plane_clouds_ds002";
boost::filesystem::path orig_file_name = "merged_plane_clouds__fusedPoses_i100_biggerLambda";

struct ObjectAnno {
    std::string object_name;
    std::vector<int> ind;
    bool is_on_floor;
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
    if (argc < 2) {
        pcl::console::print_info("\n\
                                 -- Transfer annotation -- : \n\
                                 \n\
                                 Syntax: %s annotation_folder \n\
                                 e.g. /home/edith/Annotations/Arena/",
                                 argv[0]);
        return(1);
    }

    /// Parse command line arguments
    std::string annotation_path = argv[1];

    if (!boost::filesystem::exists(annotation_path) || !boost::filesystem::is_directory(annotation_path)) {
        std::cerr << "Annotation folder does not exist " << annotation_path << std::endl;
    }


    //----------------------------read the annotation----------------------------------
    boost::filesystem::directory_iterator end_iter; // Default constructor for an iterator is the end iterator

    for (boost::filesystem::directory_iterator iter(annotation_path); iter != end_iter; ++iter) {
        if (boost::filesystem::is_directory(*iter)) {
            boost::filesystem::path planes ("planes");
            for (boost::filesystem::directory_iterator plane_iter(iter->path() / planes); plane_iter != end_iter; ++plane_iter) {
                std::cout << plane_iter->path() << std::endl;

                pcl::PointCloud<PointNormal>::Ptr ds_anno_cloud (new pcl::PointCloud<PointNormal>);
                pcl::PointCloud<PointNormal>::Ptr orig_anno_cloud (new pcl::PointCloud<PointNormal>);

                readInput((plane_iter->path() / ds_file_name).string() + ".pcd", ds_anno_cloud);
                readInput((plane_iter->path() / orig_file_name).string() + ".pcd", orig_anno_cloud);

                std::cout << "DS cloud: " << ds_anno_cloud->size() << " points" << std::endl;
                std::cout << "Orig cloud: " <<orig_anno_cloud->size() << " points" << std::endl;

                std::vector<int> orig_pts_ind;
                for (size_t i = 0; i < orig_anno_cloud->size(); i++) {
                    if (!pcl::isFinite(orig_anno_cloud->points[i])) {
                        std::cerr << "Orig cloud has NANs" << std::endl;
                        return -1;
                    }
                    orig_pts_ind.push_back(i);
                }

                /// read annotated objects
                //Read the GT.anno file for the ds scene
                std::string anno_scene_path = (plane_iter->path() / ds_file_name).string() + "_GT.anno";

                if(!boost::filesystem::exists(anno_scene_path)) {
                    std::cerr << "Couldn't find _GT.anno file for plane " << plane_iter->path().string() << ". I was looking for the file " << anno_scene_path << std::endl;
                    continue;
                }

                //read the annotation file
                std::vector<ObjectAnno> objects;
                std::ifstream anno_file(anno_scene_path);
                std::string line;
                while(std::getline(anno_file, line)) {
                    ObjectAnno obj;
                    std::string object_name;
                    std::vector<std::string> split_result;
                    boost::split(split_result, line, boost::is_any_of(" "));
                    object_name = split_result.at(0);
                    //special case airplane, because it consists of several parts
                    if (object_name.find("airplane") != std::string::npos) {
                        object_name = "airplane";
                    }
                    obj.object_name = object_name;
                    for (size_t i = 1; i < split_result.size(); i++) {
                        int id = std::atoi(split_result.at(i).c_str());
                        obj.ind.push_back(id);
                    }
//                    std::string is_on_floor = split_result[split_result.size()-1];
//                    obj.is_on_floor = (is_on_floor == "true") ? true : false;
                    objects.push_back(obj);
                }

                /// create octree for radius search
                pcl::KdTreeFLANN<PointNormal> tree;
                tree.setInputCloud(orig_anno_cloud);

                std::vector<int> nn_indices;
                std::vector<float> nn_distances;

                /// for each annoted object find corresponding points in orig cloud
                pcl::PointCloud<PointNormal>::Ptr ds_test (new pcl::PointCloud<PointNormal>);
                std::vector<ObjectAnno> orig_objects;
                for (size_t o = 0; o < objects.size(); o++) {
                    ObjectAnno orig_obj;
                    orig_obj.object_name = objects[o].object_name;
                    orig_obj.is_on_floor = objects[o].is_on_floor;
                    for (size_t i = 0; i < objects[o].ind.size(); ++i) {
                        if (!pcl::isFinite(ds_anno_cloud->points[objects[o].ind[i]]))
                            continue;

                        const PointNormal &p_object = ds_anno_cloud->points[objects[o].ind[i]];
                        ds_test->points.push_back(p_object);

                        if (tree.radiusSearch(p_object, 0.005, nn_indices, nn_distances) > 0){
                            orig_obj.ind.insert(orig_obj.ind.end(), nn_indices.begin(), nn_indices.end());
                        }
                    }
                    /// sort orig ind and remove duplicates
                    std::sort(orig_obj.ind.begin(), orig_obj.ind.end());
                    orig_obj.ind.erase(unique(orig_obj.ind.begin(), orig_obj.ind.end()), orig_obj.ind.end());

                    pcl::PointCloud<PointNormal>::Ptr test_cloud (new pcl::PointCloud<PointNormal>);
                    for (size_t i = 0; i < orig_obj.ind.size(); i++) {
                        test_cloud->points.push_back(orig_anno_cloud->points[orig_obj.ind[i]]);
                    }

                    /// only keep points that were not used before
                    std::vector<int> intersection;
                    std::set_intersection(orig_obj.ind.begin(),orig_obj.ind.end(), orig_pts_ind.begin(),orig_pts_ind.end(),
                                              std::back_inserter(intersection));
                    std::cout << (orig_obj.ind.size() - intersection.size()) << " points were already annotated" << std::endl;
                    orig_obj.ind = intersection;

                    orig_objects.push_back(orig_obj);

                    /// remove indices from available points
                    std::vector<int> diff;
                    std::set_difference(orig_pts_ind.begin(), orig_pts_ind.end(), orig_obj.ind.begin(), orig_obj.ind.end(),
                        std::inserter(diff, diff.begin()));
                    orig_pts_ind = diff;
                }

                /// write the orig objects to a txt file
                std::ofstream orig_anno_file;
                orig_anno_file.open ((plane_iter->path() / orig_file_name).string() + "_GT.anno");
                for (size_t i = 0; i < orig_objects.size(); i++) {
                    const ObjectAnno & obj = orig_objects[i];
                    orig_anno_file << obj.object_name << " ";
                    for (size_t p = 0; p < obj.ind.size(); p++) {
                        orig_anno_file << obj.ind[p] << " ";
                    }
//                    orig_anno_file << std::boolalpha << obj.is_on_floor << std::noboolalpha;
                    if (i != orig_objects.size()-1)
                        orig_anno_file << "\n";
                }
                orig_anno_file.close();
            }
        }
    }
}


