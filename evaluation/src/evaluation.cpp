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

#include <pcl/common/distances.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/filter.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>

#include <settings.h>

typedef pcl::PointXYZRGBL PointLabel;

float search_radius = 0.01;
float cluster_thr = 0.02;
int min_cluster_size = 200;

double ds_voxel_size = 0.002;

float min_det_pts = 0.5;
float min_fp_pts = 0.2; // if 1/5 of the points of an object have a wrong label, it counts as FP

std::string anno_file_name = "merged_plane_clouds_ds002";

std::string c_dis_obj_name = "curr_displaced_objects.pcd";
std::string c_new_obj_name = "curr_new_objects.pcd";
std::string c_static_obj_name = "curr_static_objects.pcd";
std::string r_dis_obj_name = "ref_displaced_objects.pcd";
std::string r_rem_obj_name = "ref_removed_objects.pcd";
std::string r_static_obj_name = "ref_static_objects.pcd";


//we also need the correspondences for static/moved objects
//NEW = 0; REMOVED = 10; STATIC = [20,30,...950], MOVED = [1000, 1010,...1950]
enum ObjectClass {STATIC = 0, MOVED = 1, REMOVED = 2, NEW = 3, UNKNOWN=4}; // STATIC = [20,30,...950], MOVED = [1000, 1010,...1950] - weird int assignment is better for visualization

std::vector<std::vector<int>> conf_mat = { { 0, 0, 0, 0, 0 },
                                           { 0, 0, 0, 0, 0 },
                                           { 0, 0, 0, 0, 0 },
                                           { 0, 0, 0, 0, 0 }}; //the last column is used to count the numbers of wrongly assigned moved objects, which can not be counted otherwise
std::vector<std::vector<int>> conf_mat_red = conf_mat;


struct SceneCompInfo {
    std::string ref_scene;
    std::string curr_scene;
    std::string result_path;
};

struct GTLabeledClouds {
    pcl::PointCloud<PointLabel>::Ptr ref_GT_cloud;
    pcl::PointCloud<PointLabel>::Ptr curr_GT_cloud;
};

struct GTObject {
    std::string name;
    pcl::PointCloud<PointLabel>::Ptr object_cloud;
    int class_label;
    GTObject(std::string _name, pcl::PointCloud<PointLabel>::Ptr _object_cloud, int _class_label=ObjectClass::UNKNOWN) :
        name(_name), object_cloud(_object_cloud), class_label(_class_label) {}
    bool is_detected = false;
};

struct Measurements {
    int nr_det_obj = 0;
    int nr_static_obj = 0;
    int nr_novel_obj = 0;
    int nr_removed_obj = 0;
    int nr_moved_obj = 0;
};

struct GTCloudsAndNumbers {
    GTLabeledClouds clouds;
    Measurements m;
    std::vector<GTObject> ref_objects;
    std::vector<GTObject> curr_objects;
};

int getMostFrequentNumber(std::vector<int> v);
int remainingClusters (pcl::PointCloud<PointLabel>::Ptr cloud, float cluster_thr, int min_cluster_size, std::string path="");
void writeSumResultsToFile(std::vector<Measurements> all_gt_results, std::vector<Measurements> all_tp_results, std::vector<Measurements> all_fp_results, std::string path);
int extractDetectedObjects(std::vector<GTObject> &gt_objects, pcl::PointCloud<PointLabel>::Ptr result_cloud, std::vector<std::string> & gt_obj, std::vector<std::string> & det_obj);
void writeObjectSummaryToFile(std::map<std::string, int> & gt_obj_count, std::map<std::string, int> & det_obj_count, std::map<std::string, int> &tp_class_obj_count, std::vector<Measurements> &all_fp_results, std::string path);
int computeFP(std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> obj_name_cloud_map, pcl::PointCloud<PointLabel>::Ptr leftover_cloud, std::string path, bool is_static=false);
pcl::PointCloud<PointLabel>::Ptr downsampleCloud(pcl::PointCloud<PointLabel>::Ptr input, double leafSize);
GTCloudsAndNumbers createGTforSceneComp(const std::map<std::string, std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> > scene_annotations_map,
                                        const SceneCompInfo &scene_comp);
void mergeVectorIntoMap (std::map<std::string, int> & global_map, std::vector<std::string> & local_vec);
std::vector<std::string> checkRemovedObjects(pcl::PointCloud<PointLabel>::Ptr removed_obj_cloud, const GTCloudsAndNumbers gt_cloud_numbers, std::map<std::string,
                                             int> &tp_class_object_count);
std::vector<std::string> checkNovelObjects(pcl::PointCloud<PointLabel>::Ptr novel_obj_cloud, const GTCloudsAndNumbers gt_cloud_numbers, std::map<std::string, int> &tp_class_object_count);
std::tuple<std::vector<std::string>, std::vector<std::string> > checkDisplacedObjects(pcl::PointCloud<PointLabel>::Ptr c_moved_obj_cloud, pcl::PointCloud<PointLabel>::Ptr r_moved_obj_cloud,
                                                                                      pcl::PointCloud<PointLabel>::Ptr c_static_obj_cloud, pcl::PointCloud<PointLabel>::Ptr r_static_obj_cloud,
                                                                                      const GTCloudsAndNumbers gt_cloud_numbers, std::map<std::string, int> &tp_class_object_count);
void writeConfMatToFile( std::vector<std::vector<int>> conf_mat, std::string path);
int categoryTransformer(int cat_label);
void computeConfusionMat(GTCloudsAndNumbers gt_objects, pcl::PointCloud<PointLabel>::Ptr ref_result_cloud,
                         pcl::PointCloud<PointLabel>::Ptr curr_result_cloud, std::vector<std::vector<int>> &conf_mat);
void relabelMatches(pcl::PointCloud<PointLabel>::Ptr ref_cloud, pcl::PointCloud<PointLabel>::Ptr curr_cloud, bool isStatic);

std::ostream& operator<<(std::ostream& output, Measurements const& m)
{
    output << "Detected objects: " << m.nr_det_obj << "\n";
    output << "Novel objects: " << m.nr_novel_obj << "\n";
    output << "Removed objects: " << m.nr_removed_obj << "\n";
    output << "Moved objects: " << m.nr_moved_obj << "\n";
    output << "Static objects: " << m.nr_static_obj << "\n";

    return output;
}

std::ostream& operator<<(std::ostream& output, std::vector<std::string> const& v)
{
    for (std::string s : v) {
        output << s << "; ";
    }

    return output;
}

bool readInput(std::string input_path, const pcl::PointCloud<PointLabel>::Ptr &cloud) {
    std::string ext = input_path.substr(input_path.length()-3, input_path.length());
    if (ext == "ply") {
        if (pcl::io::loadPLYFile<PointLabel> (input_path, *cloud) == -1)
        {
            std::cerr << "Couldn't read file " << input_path << std::endl;
            return false;
        }
        return true;
    } else if (ext == "pcd") {
        if (pcl::io::loadPCDFile<PointLabel> (input_path, *cloud) == -1)
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



float computeMeanPointDistance(pcl::PointCloud<PointLabel>::Ptr ref_object, pcl::PointCloud<PointLabel>::Ptr curr_obj) {
    PointLabel ref_p_mean, curr_p_mean;
    ref_p_mean.x = ref_p_mean.y = ref_p_mean.z = 0.0f;
    curr_p_mean = ref_p_mean;

    for (size_t i = 0; i < ref_object->size(); i++) {
        ref_p_mean.x += ref_object->points[i].x;
        ref_p_mean.y += ref_object->points[i].y;
        ref_p_mean.z += ref_object->points[i].z;
    }
    ref_p_mean.x = ref_p_mean.x/ref_object->size();
    ref_p_mean.y = ref_p_mean.y/ref_object->size();
    ref_p_mean.z = ref_p_mean.z/ref_object->size();

    for (size_t i = 0; i < curr_obj->size(); i++) {
        curr_p_mean.x += curr_obj->points[i].x;
        curr_p_mean.y += curr_obj->points[i].y;
        curr_p_mean.z += curr_obj->points[i].z;
    }
    curr_p_mean.x = curr_p_mean.x/curr_obj->size();
    curr_p_mean.y = curr_p_mean.y/curr_obj->size();
    curr_p_mean.z = curr_p_mean.z/curr_obj->size();

    return pcl::euclideanDistance(ref_p_mean, curr_p_mean);
}

int main(int argc, char* argv[])
{
    /// Check arguments and print info
    if (argc < 3) {
        pcl::console::print_info("\n\
                                 -- Evaluation of change detection between two scenes -- : \n\
                                 \n\
                                 Syntax: %s result_folder annotation_folder \n\
                                 e.g. /home/edith/Results/Arena/ /home/edith/Annotations/Arena/",
                                 argv[0]);
        return(1);
    }

    /// Parse command line arguments
    std::string result_path = argv[1];
    std::string annotation_path = argv[2];

    if (!boost::filesystem::exists(result_path) || !boost::filesystem::is_directory(result_path)) {
        std::cerr << "Result folder does not exist " << result_path << std::endl;
    }
    if (!boost::filesystem::exists(annotation_path) || !boost::filesystem::is_directory(annotation_path)) {
        std::cerr << "Annotation folder does not exist " << annotation_path << std::endl;
    }

    //----------------------------extract all scene comparisons----------------------------------

    std::vector<SceneCompInfo> scenes;
    boost::filesystem::directory_iterator end_iter; // Default constructor for an iterator is the end iterator
    for (boost::filesystem::directory_iterator iter(result_path); iter != end_iter; ++iter) {
        if (boost::filesystem::is_directory(*iter)) {
            std::cout << iter->path() << std::endl;

            SceneCompInfo scene_info;
            scene_info.result_path = iter->path().string();
            //extract the two scene names
            std::string scene_folder = iter->path().filename().string();
            size_t last_of;
            last_of = scene_folder.find_last_of("-");
            scene_info.ref_scene = scene_folder.substr(0,last_of);
            scene_info.curr_scene = scene_folder.substr(last_of+1, scene_folder.size()-1);

            scenes.push_back(scene_info);
        }
    }


    //----------------------------read the annotation----------------------------------
    std::map<std::string, int> gt_obj_count;
    std::map<std::string, std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> > scene_annotations_map; //e.g. ("scene2", ("mug", <cloud>))
    for (boost::filesystem::directory_iterator iter(annotation_path); iter != end_iter; ++iter) {
        if (boost::filesystem::is_directory(*iter)) {
            std::cout << iter->path() << std::endl;
            boost::filesystem::path planes ("planes");
            std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> objName_cloud_map;
            for (boost::filesystem::directory_iterator plane_iter(iter->path() / planes); plane_iter != end_iter; ++plane_iter) {
                std::string anno_plane_path = (plane_iter->path() / anno_file_name).string() + "_GT.anno";
                if(!boost::filesystem::exists(anno_plane_path)) {
                    std::cerr << "Couldn't find _GT.anno file for plane " << plane_iter->path().string() << ". I was looking for the file " << anno_plane_path << std::endl;
                    continue;
                }

                /// read the point cloud for the plane
                pcl::PointCloud<PointLabel>::Ptr plane_cloud(new pcl::PointCloud<PointLabel>);
                readInput((plane_iter->path() / anno_file_name).string() + ".pcd", plane_cloud);

                /// read the annotation file
                std::ifstream anno_file(anno_plane_path);
                std::string line;
                while(std::getline(anno_file, line)) {
                    pcl::PointCloud<PointLabel>::Ptr object_cloud(new pcl::PointCloud<PointLabel>);
                    std::string object_name;
                    std::vector<std::string> split_result;
                    boost::split(split_result, line, boost::is_any_of(" "));
                    object_name = split_result.at(0);
                    //special case airplane, because it consists of several parts
                    if (object_name.find("airplane") != std::string::npos) {
                        object_name = "airplane";
                    }
                    for (size_t i = 1; i < split_result.size()-1; i++) {
                        int id = std::atoi(split_result.at(i).c_str());
                        object_cloud->points.push_back(plane_cloud->points[id]);
                    }
                    object_cloud->width = object_cloud->points.size();
                    object_cloud->height = 1;
                    object_cloud->is_dense=true;
                    std::string is_on_floor = split_result[split_result.size()-1];
                    if (is_on_floor == "false") {
                        objName_cloud_map[object_name] = object_cloud;
                        std::map<std::string, int>::iterator it = gt_obj_count.find(object_name);
                        if (it == gt_obj_count.end())
                            gt_obj_count[object_name] = 1;
                        else
                            gt_obj_count[object_name] += 1;
                    }
                }
            }
            scene_annotations_map[iter->path().filename().string()] = objName_cloud_map;
        }
    }

    //----------------------------create GT for all scene comparisons from the result folder----------------------------------

    /// save scene2-scene3 (key) together with two clouds and number of detected objects
    std::map<std::string, GTCloudsAndNumbers> scene_GTClouds_map;
    for (size_t i = 0; i < scenes.size(); i++) {
        const SceneCompInfo &scene_comp = scenes[i];
        GTCloudsAndNumbers gt_cloud_numbers  = createGTforSceneComp(scene_annotations_map, scene_comp);

        if (gt_cloud_numbers.clouds.ref_GT_cloud->empty() || gt_cloud_numbers.clouds.curr_GT_cloud->empty())
            continue;

        std::string comp_string = scene_comp.ref_scene + "-" + scene_comp.curr_scene;
        scene_GTClouds_map[comp_string] = gt_cloud_numbers;

        pcl::io::savePCDFileBinary(scene_comp.result_path + "/" + scene_comp.ref_scene + "-" + scene_comp.curr_scene + ".pcd", *(gt_cloud_numbers.clouds.ref_GT_cloud));
        pcl::io::savePCDFileBinary(scene_comp.result_path + "/" + scene_comp.curr_scene + "-" + scene_comp.ref_scene + ".pcd", *(gt_cloud_numbers.clouds.curr_GT_cloud));
    }



    //----------------------------the real evaluation happens now----------------------------------
    /// iterate over all scene comparison folders
    std::vector<Measurements> all_gt_results, all_tp_results, all_fp_results;
    std::map<std::string, int> det_obj_count_comp, gt_obj_count_comp, tp_class_object_count;
    //measurements based on detected objects
    std::vector<Measurements> reduced_all_gt_results, reduced_all_tp_results, reduced_all_fp_results;
    std::map<std::string, int> reduced_det_obj_count_comp, reduced_gt_obj_count_comp, reduced_tp_class_object_count;
    for (size_t i = 0; i < scenes.size(); i++) {
        const SceneCompInfo &scene_comp = scenes[i];
        std::string scene_comp_str = scene_comp.ref_scene + "-" + scene_comp.curr_scene;
        GTCloudsAndNumbers gt_cloud_numbers = scene_GTClouds_map[scene_comp_str];

        Measurements result, FP_results;
        std::vector<std::string> ref_matched_removed, curr_matched_novel, matched_moved, matched_static;
        //measurements based on detected objects
        Measurements reduced_result, reduced_FP_results;
        std::vector<std::string> reduced_ref_matched_removed, reduced_curr_matched_novel, reduced_matched_moved, reduced_matched_static;

        pcl::PointCloud<PointLabel>::Ptr novel_obj_cloud(new pcl::PointCloud<PointLabel>);
        readInput(scene_comp.result_path + "/" + c_new_obj_name, novel_obj_cloud);
        pcl::PointCloud<PointLabel>::Ptr removed_obj_cloud(new pcl::PointCloud<PointLabel>);
        readInput(scene_comp.result_path + "/" + r_rem_obj_name, removed_obj_cloud);
        pcl::PointCloud<PointLabel>::Ptr r_moved_obj_cloud(new pcl::PointCloud<PointLabel>);
        readInput(scene_comp.result_path + "/" + r_dis_obj_name, r_moved_obj_cloud);
        pcl::PointCloud<PointLabel>::Ptr c_moved_obj_cloud(new pcl::PointCloud<PointLabel>);
        readInput(scene_comp.result_path + "/" + c_dis_obj_name, c_moved_obj_cloud);
        pcl::PointCloud<PointLabel>::Ptr r_static_obj_cloud(new pcl::PointCloud<PointLabel>);
        readInput(scene_comp.result_path + "/" + r_static_obj_name, r_static_obj_cloud);
        pcl::PointCloud<PointLabel>::Ptr c_static_obj_cloud(new pcl::PointCloud<PointLabel>);
        readInput(scene_comp.result_path + "/" + c_static_obj_name, c_static_obj_cloud);


        //HACK because I forgot to label the result clouds. i only did for static and moved objects
        for (size_t i = 0; i < novel_obj_cloud->points.size(); i++)
        {
            novel_obj_cloud->points[i].label = ObjectClass::NEW;
        }

        //HACK because I forgot to label the result clouds. i only did for static and moved objects
        for (size_t i = 0; i < removed_obj_cloud->points.size(); i++)
        {
            removed_obj_cloud->points[i].label = ObjectClass::REMOVED;
        }

        relabelMatches(r_static_obj_cloud, c_static_obj_cloud, true);
        relabelMatches(r_moved_obj_cloud, c_moved_obj_cloud, false);

        //combine all detected results and compare it to GT
        std::vector<std::string> curr_gt_obj_vec, curr_det_obj_vec, ref_gt_obj_vec, ref_det_obj_vec;
        pcl::PointCloud<PointLabel>::Ptr curr_result_cloud(new pcl::PointCloud<PointLabel>());
        *curr_result_cloud += *c_moved_obj_cloud;
        *curr_result_cloud += *c_static_obj_cloud;
        *curr_result_cloud += *novel_obj_cloud;

        if (!curr_result_cloud->empty())
            pcl::io::savePCDFileBinary( scene_comp.result_path+"/curr_result_cloud.pcd", *curr_result_cloud);
        result.nr_det_obj = extractDetectedObjects(gt_cloud_numbers.curr_objects, curr_result_cloud, curr_gt_obj_vec, curr_det_obj_vec);
        mergeVectorIntoMap(gt_obj_count_comp, curr_gt_obj_vec);
        mergeVectorIntoMap(det_obj_count_comp, curr_det_obj_vec);
        std::sort(curr_det_obj_vec.begin(), curr_det_obj_vec.end());


        pcl::PointCloud<PointLabel>::Ptr ref_result_cloud(new pcl::PointCloud<PointLabel>());
        *ref_result_cloud += *r_moved_obj_cloud;
        *ref_result_cloud += *r_static_obj_cloud;
        *ref_result_cloud += *removed_obj_cloud;
        if (!ref_result_cloud->empty())
            pcl::io::savePCDFileBinary( scene_comp.result_path+"/ref_result_cloud.pcd", *ref_result_cloud);
        result.nr_det_obj += extractDetectedObjects(gt_cloud_numbers.ref_objects, ref_result_cloud, ref_gt_obj_vec, ref_det_obj_vec);
        mergeVectorIntoMap(gt_obj_count_comp, ref_gt_obj_vec);
        mergeVectorIntoMap(det_obj_count_comp, ref_det_obj_vec);
        std::sort(ref_det_obj_vec.begin(), ref_det_obj_vec.end());


        //prepare GT based on detected objects
        std::map<std::string, std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> > reduced_scene_annotations_map;
        std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> curr_scene_GT_obj = scene_annotations_map[scene_comp.curr_scene];
        std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> ref_scene_GT_obj = scene_annotations_map[scene_comp.ref_scene];
        std::sort(curr_gt_obj_vec.begin(), curr_gt_obj_vec.end());
        std::sort(curr_det_obj_vec.begin(), curr_det_obj_vec.end());
        std::sort(ref_gt_obj_vec.begin(), ref_gt_obj_vec.end());
        std::sort(ref_det_obj_vec.begin(), ref_det_obj_vec.end());
        std::vector<std::string> diff;
        //objects that were not detected
        std::set_difference(curr_gt_obj_vec.begin(), curr_gt_obj_vec.end(), curr_det_obj_vec.begin(), curr_det_obj_vec.end(), std::inserter(diff, diff.begin()));
        //remove them from the annotation map
        for (std::string not_det_obj : diff) {
            curr_scene_GT_obj.erase(not_det_obj);
        }
        diff.clear();
        std::set_difference(ref_gt_obj_vec.begin(), ref_gt_obj_vec.end(), ref_det_obj_vec.begin(), ref_det_obj_vec.end(), std::inserter(diff, diff.begin()));
        //remove them from the annotation map
        for (std::string not_det_obj : diff) {
            ref_scene_GT_obj.erase(not_det_obj);
        }
        reduced_scene_annotations_map[scene_comp.curr_scene] = curr_scene_GT_obj;
        reduced_scene_annotations_map[scene_comp.ref_scene] = ref_scene_GT_obj;

        GTCloudsAndNumbers reduced_gt_cloud_numbers  = createGTforSceneComp(reduced_scene_annotations_map, scene_comp);
        pcl::io::savePCDFileBinary(scene_comp.result_path + "/reduced_" + scene_comp.ref_scene + "-" + scene_comp.curr_scene + ".pcd", *(reduced_gt_cloud_numbers.clouds.ref_GT_cloud));
        pcl::io::savePCDFileBinary(scene_comp.result_path + "/reduced_" + scene_comp.curr_scene + "-" + scene_comp.ref_scene + ".pcd", *(reduced_gt_cloud_numbers.clouds.curr_GT_cloud));

        std::vector<std::string> reduced_curr_gt_obj_vec, reduced_curr_det_obj_vec, reduced_ref_gt_obj_vec, reduced_ref_det_obj_vec;
        reduced_result.nr_det_obj = extractDetectedObjects(reduced_gt_cloud_numbers.curr_objects, curr_result_cloud, reduced_curr_gt_obj_vec, reduced_curr_det_obj_vec);
        mergeVectorIntoMap(reduced_gt_obj_count_comp, reduced_curr_gt_obj_vec);
        mergeVectorIntoMap(reduced_det_obj_count_comp, reduced_curr_det_obj_vec);
        reduced_result.nr_det_obj += extractDetectedObjects(reduced_gt_cloud_numbers.ref_objects, ref_result_cloud, reduced_ref_gt_obj_vec, reduced_ref_det_obj_vec);
        mergeVectorIntoMap(reduced_gt_obj_count_comp, reduced_ref_gt_obj_vec);
        mergeVectorIntoMap(reduced_det_obj_count_comp, reduced_ref_det_obj_vec);

        computeConfusionMat(gt_cloud_numbers, ref_result_cloud, curr_result_cloud, conf_mat);
        computeConfusionMat(reduced_gt_cloud_numbers, ref_result_cloud, curr_result_cloud, conf_mat_red);

        /// Check REMOVED objects
        if (!removed_obj_cloud->empty()) {
            pcl::PointCloud<PointLabel>::Ptr removed_obj_cloud_copy(new pcl::PointCloud<PointLabel>);
            pcl::copyPointCloud(*removed_obj_cloud, *removed_obj_cloud_copy);

            std::vector<std::string> corr_obj_str = checkRemovedObjects(removed_obj_cloud_copy, gt_cloud_numbers, tp_class_object_count);
            ref_matched_removed.insert(ref_matched_removed.end(), corr_obj_str.begin(), corr_obj_str.end());
            result.nr_removed_obj += corr_obj_str.size();

            FP_results.nr_removed_obj = computeFP(scene_annotations_map[scene_comp.ref_scene], removed_obj_cloud_copy, scene_comp.result_path+"/rem_removed");

            //based on detected objects
            pcl::copyPointCloud(*removed_obj_cloud, *removed_obj_cloud_copy);
            corr_obj_str = checkRemovedObjects(removed_obj_cloud_copy, reduced_gt_cloud_numbers, reduced_tp_class_object_count);
            reduced_ref_matched_removed.insert(reduced_ref_matched_removed.end(), corr_obj_str.begin(), corr_obj_str.end());
            reduced_result.nr_removed_obj += corr_obj_str.size();
            reduced_FP_results.nr_removed_obj = computeFP(reduced_scene_annotations_map[scene_comp.ref_scene], removed_obj_cloud_copy, scene_comp.result_path+"/rem_reduced_removed");
        }
        /// ------------------------------------------------------------------------------------------------------------------------------------------------------------------

        /// Check NOVEL objects
        if (!novel_obj_cloud->empty()) {
            pcl::PointCloud<PointLabel>::Ptr novel_obj_cloud_copy(new pcl::PointCloud<PointLabel>);
            pcl::copyPointCloud(*novel_obj_cloud, *novel_obj_cloud_copy);

            std::vector<std::string> corr_obj_str = checkNovelObjects(novel_obj_cloud_copy, gt_cloud_numbers, tp_class_object_count);
            curr_matched_novel.insert(curr_matched_novel.end(), corr_obj_str.begin(), corr_obj_str.end());
            result.nr_novel_obj += corr_obj_str.size();

            FP_results.nr_novel_obj = computeFP(scene_annotations_map[scene_comp.curr_scene], novel_obj_cloud_copy, scene_comp.result_path+"/rem_novel");

            //based on detected objects
            pcl::copyPointCloud(*novel_obj_cloud, *novel_obj_cloud_copy);

            corr_obj_str = checkNovelObjects(novel_obj_cloud_copy, reduced_gt_cloud_numbers, reduced_tp_class_object_count);
            reduced_curr_matched_novel.insert(reduced_curr_matched_novel.end(), corr_obj_str.begin(), corr_obj_str.end());
            reduced_result.nr_novel_obj += corr_obj_str.size();

            reduced_FP_results.nr_novel_obj = computeFP(reduced_scene_annotations_map[scene_comp.curr_scene], novel_obj_cloud_copy, scene_comp.result_path+"/rem_reduced_novel");
        }
        /// ------------------------------------------------------------------------------------------------------------------------------------------------------------------


        /// Check MOVED and STATIC objects
        pcl::PointCloud<PointLabel>::Ptr c_moved_obj_cloud_copy(new pcl::PointCloud<PointLabel>);
        pcl::copyPointCloud(*c_moved_obj_cloud, *c_moved_obj_cloud_copy);
        pcl::PointCloud<PointLabel>::Ptr r_moved_obj_cloud_copy(new pcl::PointCloud<PointLabel>);
        pcl::copyPointCloud(*r_moved_obj_cloud, *r_moved_obj_cloud_copy);
        pcl::PointCloud<PointLabel>::Ptr c_static_obj_cloud_copy(new pcl::PointCloud<PointLabel>);
        pcl::copyPointCloud(*c_static_obj_cloud, *c_static_obj_cloud_copy);
        pcl::PointCloud<PointLabel>::Ptr r_static_obj_cloud_copy(new pcl::PointCloud<PointLabel>);
        pcl::copyPointCloud(*r_static_obj_cloud, *r_static_obj_cloud_copy);

        std::tuple<std::vector<std::string>, std::vector<std::string>> matched_ojects = checkDisplacedObjects(c_moved_obj_cloud_copy, r_moved_obj_cloud_copy,
                                                                                                              c_static_obj_cloud_copy, r_static_obj_cloud_copy,
                                                                                                              gt_cloud_numbers, tp_class_object_count);

        matched_moved.insert(matched_moved.end(), std::get<1>(matched_ojects).begin(),  std::get<1>(matched_ojects).end());
        result.nr_moved_obj += (2 * std::get<1>(matched_ojects).size());
        matched_static.insert(matched_static.end(),  std::get<0>(matched_ojects).begin(),  std::get<0>(matched_ojects).end());
        result.nr_static_obj += (2 * std::get<0>(matched_ojects).size());


        FP_results.nr_moved_obj = computeFP(scene_annotations_map[scene_comp.ref_scene], r_moved_obj_cloud_copy, scene_comp.result_path+"/rem_ref_moved");
        FP_results.nr_moved_obj += computeFP(scene_annotations_map[scene_comp.curr_scene], c_moved_obj_cloud_copy, scene_comp.result_path+"/rem_curr_moved");

        //TODO: is there something like static FP? everything in the scene is more or less static except for YCB objects
        //            FP_results.nr_static_obj = remainingClusters(r_static_obj_cloud, cluster_thr, min_cluster_size, scene_comp.result_path+"/rem_ref_static.pcd");
        //            FP_results.nr_static_obj += remainingClusters(c_static_obj_cloud, cluster_thr, min_cluster_size, scene_comp.result_path+"/rem_curr_static.pcd");
        FP_results.nr_static_obj = computeFP(scene_annotations_map[scene_comp.ref_scene], r_static_obj_cloud_copy, scene_comp.result_path+"/rem_ref_static", true);
        FP_results.nr_static_obj += computeFP(scene_annotations_map[scene_comp.curr_scene], c_static_obj_cloud_copy, scene_comp.result_path+"/rem_curr_static", true);
        //        if (!r_static_obj_cloud->empty())
        //            pcl::io::savePCDFileBinary(scene_comp.result_path+"/rem_ref_static.pcd", *r_static_obj_cloud);
        //        if (!c_static_obj_cloud->empty())
        //            pcl::io::savePCDFileBinary( scene_comp.result_path+"/rem_curr_static.pcd", *c_static_obj_cloud);

        //based on detected objects
        pcl::copyPointCloud(*c_moved_obj_cloud, *c_moved_obj_cloud_copy);
        pcl::copyPointCloud(*r_moved_obj_cloud, *r_moved_obj_cloud_copy);
        pcl::copyPointCloud(*c_static_obj_cloud, *c_static_obj_cloud_copy);
        pcl::copyPointCloud(*r_static_obj_cloud, *r_static_obj_cloud_copy);

        matched_ojects = checkDisplacedObjects(c_moved_obj_cloud_copy, r_moved_obj_cloud_copy, c_static_obj_cloud_copy, r_static_obj_cloud_copy,
                                               reduced_gt_cloud_numbers, reduced_tp_class_object_count);

        reduced_matched_moved.insert(reduced_matched_moved.end(), std::get<1>(matched_ojects).begin(),  std::get<1>(matched_ojects).end());
        reduced_result.nr_moved_obj += (2 * std::get<1>(matched_ojects).size());
        reduced_matched_static.insert(reduced_matched_static.end(),  std::get<0>(matched_ojects).begin(),  std::get<0>(matched_ojects).end());
        reduced_result.nr_static_obj += (2 * std::get<0>(matched_ojects).size());


        reduced_FP_results.nr_moved_obj = computeFP(reduced_scene_annotations_map[scene_comp.ref_scene], r_moved_obj_cloud_copy, scene_comp.result_path+"/rem_reduced_ref_moved");
        reduced_FP_results.nr_moved_obj += computeFP(reduced_scene_annotations_map[scene_comp.curr_scene], c_moved_obj_cloud_copy, scene_comp.result_path+"/rem_reduced_curr_moved");

        //TODO: is there something like static FP? everything in the scene is more or less static except for YCB objects
        //            FP_results.nr_static_obj = remainingClusters(r_static_obj_cloud, cluster_thr, min_cluster_size, scene_comp.result_path+"/rem_ref_static.pcd");
        //            FP_results.nr_static_obj += remainingClusters(c_static_obj_cloud, cluster_thr, min_cluster_size, scene_comp.result_path+"/rem_curr_static.pcd");
        reduced_FP_results.nr_static_obj = computeFP(reduced_scene_annotations_map[scene_comp.ref_scene], r_static_obj_cloud_copy, scene_comp.result_path+"/rem_reduced_ref_static", true);
        reduced_FP_results.nr_static_obj += computeFP(reduced_scene_annotations_map[scene_comp.curr_scene], c_static_obj_cloud_copy, scene_comp.result_path+"/rem_reduced_curr_static", true);
        //        if (!r_static_obj_cloud->empty())
        //            pcl::io::savePCDFileBinary(scene_comp.result_path+"/rem_reduced_ref_static.pcd", *r_static_obj_cloud);
        //        if (!c_static_obj_cloud->empty())
        //            pcl::io::savePCDFileBinary( scene_comp.result_path+"/rem_reduced_curr_static.pcd", *c_static_obj_cloud);

        /// ------------------------------------------------------------------------------------------------------------------------------------------------------------------



        std::cout << scene_comp.ref_scene + "-" + scene_comp.curr_scene << std::endl;
        std::cout << "GT numbers \n" << gt_cloud_numbers.m;
        std::cout << "-------------------------------------------" << std::endl;
        std::cout << "True Positives" << "\n" << result;
        std::cout << "-------------------------------------------" << std::endl;
        std::cout << "False Positives" << "\n" << FP_results;
        std::cout << "-------------------------------------------" << std::endl;
        std::cout << "Found NOVEL objects: " << curr_matched_novel << std::endl;
        std::cout << "Found REMOVED objects: " << ref_matched_removed << std::endl;
        std::cout << "Found MOVED objects: " << matched_moved << std::endl;
        std::cout << "Found STATIC objects: " << matched_static << std::endl;
        std::cout << "###############################################" << std::endl;

        std::ofstream scene_comp_result_file;
        scene_comp_result_file.open(scene_comp.result_path+"/result.txt");
        scene_comp_result_file << "GT numbers \n" << gt_cloud_numbers.m;
        scene_comp_result_file << "-------------------------------------------" << "\n";
        scene_comp_result_file << "True Positives" << "\n" << result;
        scene_comp_result_file << "-------------------------------------------" << "\n";
        scene_comp_result_file << "False Positives" << "\n" << FP_results;
        scene_comp_result_file << "-------------------------------------------" << "\n";
        scene_comp_result_file << "Found NOVEL objects: " << curr_matched_novel << "\n";
        scene_comp_result_file << "Found REMOVED objects: " << ref_matched_removed << "\n";
        scene_comp_result_file << "Found MOVED objects: " << matched_moved << "\n";
        scene_comp_result_file << "Found STATIC objects: " << matched_static << "\n";

        scene_comp_result_file << "\n-------------------------------Based on detected objects----------------------------------------\n";
        scene_comp_result_file << "GT numbers \n" << reduced_gt_cloud_numbers.m;
        scene_comp_result_file << "-------------------------------------------" << "\n";
        scene_comp_result_file << "True Positives" << "\n" << reduced_result;
        scene_comp_result_file << "-------------------------------------------" << "\n";
        scene_comp_result_file << "False Positives" << "\n" << reduced_FP_results;
        scene_comp_result_file << "-------------------------------------------" << "\n";
        scene_comp_result_file << "Found NOVEL objects: " << reduced_curr_matched_novel << "\n";
        scene_comp_result_file << "Found REMOVED objects: " << reduced_ref_matched_removed << "\n";
        scene_comp_result_file << "Found MOVED objects: " << reduced_matched_moved << "\n";
        scene_comp_result_file << "Found STATIC objects: " << reduced_matched_static << "\n";
        scene_comp_result_file.close();

        all_gt_results.push_back(gt_cloud_numbers.m);
        all_tp_results.push_back(result);
        all_fp_results.push_back(FP_results);

        reduced_all_gt_results.push_back(reduced_gt_cloud_numbers.m);
        reduced_all_tp_results.push_back(reduced_result);
        reduced_all_fp_results.push_back(reduced_FP_results);

    }

    std::cout << "Overview of objects used in the scenes" << std::endl; //sum of objects used in scene2-scene6
    for (const auto obj_count : gt_obj_count) {
        std::cout << obj_count.first << ": " << obj_count.second << std::endl;
    }

    std::cout << "Number of GT objects using all possible scene comparisons" << std::endl; //sum of objects used in scene2-scene6
    for (const auto obj_count : gt_obj_count_comp) {
        std::cout << obj_count.first << ": " << obj_count.second << std::endl;
    }

    std::cout << "Number of detected objects using all possible scene comparisons" << std::endl; //sum of objects used in scene2-scene6
    for (const auto obj_count : det_obj_count_comp) {
        std::cout << obj_count.first << ": " << obj_count.second << std::endl;
    }

    std::string result_file = result_path +  "/results.txt";
    //clear the file content
    std::ofstream result_stream (result_file);
    result_stream.close();
    writeSumResultsToFile(all_gt_results, all_tp_results, all_fp_results, result_file);
    writeObjectSummaryToFile(gt_obj_count_comp, det_obj_count_comp, tp_class_object_count, all_fp_results, result_file);
    writeConfMatToFile(conf_mat, result_file);

    //write results based on detected objects
    result_stream.open (result_file, std::ofstream::out | std::ofstream::app);
    result_stream << "\n\n-------------------------------Based on detected objects----------------------------------------\n";
    result_stream.close();
    writeSumResultsToFile(reduced_all_gt_results, reduced_all_tp_results, reduced_all_fp_results, result_file);
    writeObjectSummaryToFile(reduced_gt_obj_count_comp, reduced_det_obj_count_comp, reduced_tp_class_object_count, reduced_all_fp_results, result_file);
    writeConfMatToFile(conf_mat_red, result_file);

}

int remainingClusters (pcl::PointCloud<PointLabel>::Ptr cloud, float cluster_thr, int min_cluster_size, std::string path) {
    //clean up small things
    if (cloud->empty()) {
        return 0;
    }
    //check if cloud only consists of nans
    std::vector<int> nan_ind;
    pcl::PointCloud<PointLabel>::Ptr no_nans_cloud(new pcl::PointCloud<PointLabel>);
    cloud->is_dense = false;
    pcl::removeNaNFromPointCloud(*cloud, *no_nans_cloud, nan_ind);
    if (no_nans_cloud->size() == 0) {
        return 0;
    }

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::search::KdTree<PointLabel>::Ptr tree (new pcl::search::KdTree<PointLabel>);
    tree->setInputCloud (no_nans_cloud);

    pcl::EuclideanClusterExtraction<PointLabel> ec;
    ec.setClusterTolerance (cluster_thr);
    ec.setMinClusterSize (min_cluster_size);
    ec.setMaxClusterSize (std::numeric_limits<int>::max());
    ec.setSearchMethod(tree);
    ec.setInputCloud (no_nans_cloud);
    ec.extract (cluster_indices);

    //extract remaining cloud
    std::vector<int> cluster_ind;
    for (size_t i = 0; i < cluster_indices.size(); i++) {
        cluster_ind.insert(std::end(cluster_ind), std::begin(cluster_indices.at(i).indices), std::end(cluster_indices.at(i).indices));
    }
    pcl::ExtractIndices<PointLabel> extract;
    pcl::PointIndices::Ptr c_ind(new pcl::PointIndices());
    extract.setInputCloud (no_nans_cloud);
    c_ind->indices = cluster_ind;
    extract.setIndices(c_ind);
    extract.setKeepOrganized(false);
    extract.setNegative (false);
    extract.filter(*no_nans_cloud);

    if(!no_nans_cloud->empty() && path != "")
        pcl::io::savePCDFile(path, *no_nans_cloud);

    return cluster_indices.size();
}

int computeFP(std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> obj_name_cloud_map, pcl::PointCloud<PointLabel>::Ptr leftover_cloud, std::string path, bool is_static) {
    int fp = 0;
    std::vector<int> nan_ind;
    pcl::PointCloud<PointLabel>::Ptr no_nans_cloud(new pcl::PointCloud<PointLabel>);
    leftover_cloud->is_dense = false;
    pcl::removeNaNFromPointCloud(*leftover_cloud, *no_nans_cloud, nan_ind);
    if (no_nans_cloud->size() == 0) {
        return 0;
    }
    std::vector<int> all_gt_overlapping_ind;
    for (auto gt_obj : obj_name_cloud_map) {
        int nr_det_obj = 0;
        pcl::search::KdTree<PointLabel>::Ptr tree (new pcl::search::KdTree<PointLabel>);
        tree->setInputCloud (no_nans_cloud);
        std::vector<int> nn_indices, overlapping_ind;
        std::vector<float> nn_distances;

        for (const PointLabel p : gt_obj.second->points) {
            if (tree->radiusSearch(p, 0.005, nn_indices, nn_distances) > 0) {
                overlapping_ind.insert(overlapping_ind.end(), nn_indices.begin(), nn_indices.end());
            }
        }
        std::sort(overlapping_ind.begin(), overlapping_ind.end());
        overlapping_ind.erase(std::unique( overlapping_ind.begin(), overlapping_ind.end() ), overlapping_ind.end() );
        //if more than 1/5 of the GT object
        if (overlapping_ind.size() > gt_obj.second->size() * min_fp_pts) {
            fp++;
            std::cout << "FP: " << gt_obj.first << std::endl;
            all_gt_overlapping_ind.insert(all_gt_overlapping_ind.begin(), overlapping_ind.begin(), overlapping_ind.end());
        }
    }

    std::sort(all_gt_overlapping_ind.begin(), all_gt_overlapping_ind.end());
    all_gt_overlapping_ind.erase(std::unique( all_gt_overlapping_ind.begin(), all_gt_overlapping_ind.end() ), all_gt_overlapping_ind.end() );

    //remove overlapping ind from the cloud
    pcl::PointCloud<PointLabel>::Ptr FP_GT_obj_cloud(new pcl::PointCloud<PointLabel>);
    pcl::PointIndices::Ptr c_ind(new pcl::PointIndices());
    c_ind->indices = all_gt_overlapping_ind;
    pcl::ExtractIndices<PointLabel> extract;
    extract.setInputCloud (no_nans_cloud);
    extract.setIndices(c_ind);
    extract.setKeepOrganized(false);
    extract.setNegative (false);
    extract.filter(*FP_GT_obj_cloud);

    if (!FP_GT_obj_cloud->empty())
        pcl::io::savePCDFileBinary(path + "_gt_obj.pcd", *FP_GT_obj_cloud);

    //there is nothing like FP for static objects when it is not an object from the GT
    if (!is_static) {
        extract.setNegative (true);
        extract.filter(*no_nans_cloud);

        fp += remainingClusters(no_nans_cloud, cluster_thr, min_cluster_size, path + "_other_obj.pcd");
    }

    return fp;
}

//an object counts as detected if > min_det_pts of it was detected
//assuming there is one object instance in the scene!
int extractDetectedObjects(std::vector<GTObject> &gt_objects, pcl::PointCloud<PointLabel>::Ptr result_cloud,
                           std::vector<std::string> & gt_obj_vec, std::vector<std::string> & det_obj_vec) {
    int nr_det_obj = 0;
    pcl::search::KdTree<PointLabel>::Ptr tree (new pcl::search::KdTree<PointLabel>);
    tree->setInputCloud (result_cloud);
    std::vector<int> nn_indices, overlapping_ind;
    std::vector<float> nn_distances;

    for (GTObject & gt_obj : gt_objects) {
        pcl::PointCloud<PointLabel>::Ptr gt_cloud = gt_obj.object_cloud;
        overlapping_ind.clear();
        for (const PointLabel p : gt_cloud->points) {
            if (tree->radiusSearch(p, 0.01, nn_indices, nn_distances) > 0) {
                overlapping_ind.insert(overlapping_ind.begin(), nn_indices.begin(), nn_indices.end());
            }
        }

        std::sort(overlapping_ind.begin(), overlapping_ind.end());
        overlapping_ind.erase(unique(overlapping_ind.begin(), overlapping_ind.end()), overlapping_ind.end());
        if (overlapping_ind.size() > min_det_pts * gt_cloud->size()) {
            nr_det_obj++;
            det_obj_vec.push_back(gt_obj.name);
            gt_obj.is_detected = true;
        }
        gt_obj_vec.push_back(gt_obj.name);
    }
    return nr_det_obj;
}

void mergeVectorIntoMap (std::map<std::string, int> & global_map, std::vector<std::string> & local_vec) {
    for (size_t i=0; i < local_vec.size(); i++) {
        std::map<std::string, int>::iterator it = global_map.find(local_vec[i]);
        if (it == global_map.end())
            global_map[local_vec[i]] = 1;
        else
            global_map[local_vec[i]] += 1;
    }
}

void writeObjectSummaryToFile(std::map<std::string, int> & gt_obj_count, std::map<std::string, int> & det_obj_count, std::map<std::string, int> & tp_class_obj_count,
                              std::vector<Measurements> &all_fp_results, std::string path) {
    int count_gt_obj = 0;
    int count_det_obj = 0;
    int count_tp_class_obj = 0;
    std::ofstream result_file;
    result_file.open (path, std::ofstream::out | std::ofstream::app);
    result_file << "\n";
    for (const auto obj_count : gt_obj_count) {
        const std::string &obj_name = obj_count.first;
        result_file << obj_count.first << ": " << obj_count.second;
        count_gt_obj += obj_count.second;
        std::map<std::string, int>::iterator det_it = det_obj_count.find(obj_name);
        if (det_it != det_obj_count.end()) {
            result_file << "/" << det_it->second;
            count_det_obj += det_it->second;
        } else {
            result_file << "/" << 0;
        }
        det_it = tp_class_obj_count.find(obj_name);
        if (det_it != tp_class_obj_count.end()) {
            result_file << "/" << det_it->second;
            count_tp_class_obj += det_it->second;
        } else {
            result_file << "/" << 0;
        }
        result_file << "\n";
    }

    //sum of FP
    int FP_sum=0;
    for (Measurements m : all_fp_results) {
        FP_sum += m.nr_moved_obj;
        FP_sum += m.nr_novel_obj;
        FP_sum += m.nr_removed_obj;
        FP_sum += m.nr_static_obj;
    }

    result_file << "#objects in scene: " << count_gt_obj << "\n";
    result_file << "#detected objects: " << count_det_obj << "\n";
    result_file << "#objects correctly classified: " << count_tp_class_obj << "\n";
    result_file << "#objects wrongly classified: " << FP_sum << "\n";
    result_file.close();
}

void writeSumResultsToFile(std::vector<Measurements> all_gt_results, std::vector<Measurements> all_tp_results, std::vector<Measurements> all_fp_results,
                           std::string path) {
    Measurements gt_sum, tp_sum, fp_sum;
    for (Measurements m : all_gt_results) {
        gt_sum.nr_det_obj += m.nr_det_obj;
        gt_sum.nr_moved_obj += m.nr_moved_obj;
        gt_sum.nr_novel_obj += m.nr_novel_obj;
        gt_sum.nr_removed_obj += m.nr_removed_obj;
        gt_sum.nr_static_obj += m.nr_static_obj;
    }
    for (Measurements m : all_tp_results) {
        tp_sum.nr_det_obj += m.nr_det_obj;
        tp_sum.nr_moved_obj += m.nr_moved_obj;
        tp_sum.nr_novel_obj += m.nr_novel_obj;
        tp_sum.nr_removed_obj += m.nr_removed_obj;
        tp_sum.nr_static_obj += m.nr_static_obj;
    }
    for (Measurements m : all_fp_results) {
        fp_sum.nr_det_obj += m.nr_det_obj;
        fp_sum.nr_moved_obj += m.nr_moved_obj;
        fp_sum.nr_novel_obj += m.nr_novel_obj;
        fp_sum.nr_removed_obj += m.nr_removed_obj;
        fp_sum.nr_static_obj += m.nr_static_obj;
    }

    std::ofstream result_file;
    result_file.open (path, std::ofstream::out | std::ofstream::app);
    result_file << "Ground Truth \n" << gt_sum;
    result_file << "\nTrue Positives \n" << tp_sum;
    result_file << "\nFalse Positives \n" << fp_sum;
    result_file.close();
}

void writeConfMatToFile( std::vector<std::vector<int>> conf_mat, std::string path) {
    std::ofstream result_file;
    result_file.open (path, std::ofstream::out | std::ofstream::app);
    result_file << "\n";

    result_file << "\t" << "s" << "\t" << "m" << "\t" << "r" << "\t" << "n" << "\t" << "false_ass" << "\n";
    result_file << "s\t" << conf_mat[0][0] << "\t" << conf_mat[0][1] << "\t" << conf_mat[0][2] << "\t" << conf_mat[0][3] << "\t" << conf_mat[0][4] << "\n";
    result_file << "m\t" << conf_mat[1][0] << "\t" << conf_mat[1][1] << "\t" << conf_mat[1][2] << "\t" << conf_mat[1][3] << "\t" << conf_mat[1][4] << "\n";
    result_file << "r\t" << conf_mat[2][0] << "\t" << conf_mat[2][1] << "\t" << conf_mat[2][2] << "\t" << conf_mat[2][3] << "\t" << conf_mat[2][4] << "\n";
    result_file << "n\t" << conf_mat[3][0] << "\t" << conf_mat[3][1] << "\t" << conf_mat[3][2] << "\t" << conf_mat[3][3] << "\t" << conf_mat[3][4] << "\n";

    result_file.close();
}

int getMostFrequentNumber(std::vector<int> v) {
    int maxCount = 0, mostElement = *(v.begin());
    int sz = v.size(); // to avoid calculating the size every time
    for(int i=0; i < sz; i++)
    {
        int c = count(v.begin(), v.end(), v.at(i));
        if(c > maxCount)
        {   maxCount = c;
            mostElement = v.at(i);
        }
    }
    return mostElement;
}

pcl::PointCloud<PointLabel>::Ptr downsampleCloud(pcl::PointCloud<PointLabel>::Ptr input, double leafSize)
{
    std::cout << "PointCloud before filtering has: " << input->points.size () << " data points." << std::endl;

    // Create the filtering object: downsample the dataset using a leaf size
    pcl::VoxelGrid<PointLabel> vg;
    pcl::PointCloud<PointLabel>::Ptr cloud_filtered (new pcl::PointCloud<PointLabel>);
    vg.setInputCloud (input);
    vg.setLeafSize (leafSize, leafSize, leafSize);
    vg.setDownsampleAllData(true);
    vg.filter (*cloud_filtered);
    std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl;

    return cloud_filtered;
}

GTCloudsAndNumbers createGTforSceneComp(const std::map<std::string, std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> > scene_annotations_map,
                                        const SceneCompInfo &scene_comp) {
    GTCloudsAndNumbers gt_cloud_numbers;
    //get the corresponding annotated clouds
    std::map<std::string, std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> >::const_iterator scene_anno_it_ref = scene_annotations_map.find(scene_comp.ref_scene);
    if (scene_anno_it_ref == scene_annotations_map.end()) {
        std::cerr << "Couldn't find the GT in the map for scene " << scene_comp.ref_scene << std::endl;
        return gt_cloud_numbers;
    }
    std::map<std::string, std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> >::const_iterator scene_anno_it_curr = scene_annotations_map.find(scene_comp.curr_scene);
    if (scene_anno_it_curr == scene_annotations_map.end()) {
        std::cerr << "Couldn't find the GT in the map for scene " << scene_comp.curr_scene << std::endl;
        return gt_cloud_numbers;
    }

    const std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> &ref_objName_cloud_map = scene_anno_it_ref->second;
    const std::map<std::string, pcl::PointCloud<PointLabel>::Ptr> &curr_objName_cloud_map = scene_anno_it_curr->second;

    //merge all objects in a labeled cloud depending if it is new/removed/moved/static
    pcl::PointCloud<PointLabel>::Ptr ref_GT_cloud(new pcl::PointCloud<PointLabel>);
    pcl::PointCloud<PointLabel>::Ptr curr_GT_cloud(new pcl::PointCloud<PointLabel>);

    Measurements gt_measurements;
    std::vector<GTObject> ref_gt_objects;
    std::vector<GTObject> curr_gt_objects;

    uint static_cnt = 20;
    uint moved_cnt = 1000;
    // go through all objects of the ref scene and classify them
    std::map<std::string, pcl::PointCloud<PointLabel>::Ptr>::const_iterator ref_map_it;
    for (ref_map_it = ref_objName_cloud_map.begin(); ref_map_it != ref_objName_cloud_map.end(); ref_map_it++) {
        const std::string &ref_obj_name = ref_map_it->first;
        std::map<std::string, pcl::PointCloud<PointLabel>::Ptr>::const_iterator curr_map_it = curr_objName_cloud_map.find(ref_obj_name);
        if (curr_map_it == curr_objName_cloud_map.end()) { //ref object removed
            for (size_t i = 0; i < ref_map_it->second->points.size(); i++) {
                ref_map_it->second->points[i].label = ObjectClass::REMOVED;
            }
            gt_measurements.nr_removed_obj++;
            ref_gt_objects.push_back(GTObject(ref_obj_name, ref_map_it->second, ObjectClass::REMOVED));
        } else { //check if it is either static or moved. label also the corresponding curr objects
            float obj_dist = computeMeanPointDistance(ref_map_it->second, curr_map_it->second);
            if (obj_dist < max_dist_for_being_static) {
                for (size_t i = 0; i < ref_map_it->second->points.size(); i++) {
                    ref_map_it->second->points[i].label = static_cnt; //ObjectClass::STATIC;
                }
                for (size_t i = 0; i < curr_map_it->second->points.size(); i++) {
                    curr_map_it->second->points[i].label = static_cnt; //ObjectClass::STATIC;
                }
                ref_gt_objects.push_back(GTObject(ref_obj_name, ref_map_it->second, static_cnt));
                curr_gt_objects.push_back(GTObject(curr_map_it->first, curr_map_it->second, static_cnt));
                gt_measurements.nr_static_obj+=2;
                static_cnt += 10;
            } else {
                for (size_t i = 0; i < ref_map_it->second->points.size(); i++) {
                    ref_map_it->second->points[i].label = moved_cnt; //ObjectClass::MOVED;
                }
                for (size_t i = 0; i < curr_map_it->second->points.size(); i++) {
                    curr_map_it->second->points[i].label = moved_cnt; //ObjectClass::MOVED;
                }
                ref_gt_objects.push_back(GTObject(ref_obj_name, ref_map_it->second, moved_cnt));
                curr_gt_objects.push_back(GTObject(curr_map_it->first, curr_map_it->second, moved_cnt));
                moved_cnt += 10;
                gt_measurements.nr_moved_obj+=2;
            }
            *curr_GT_cloud += *(curr_map_it->second);
        }
        *ref_GT_cloud += *(ref_map_it->second);
    }


    // go through all objects of the curr scene and classify them
    std::map<std::string, pcl::PointCloud<PointLabel>::Ptr>::const_iterator curr_map_it;
    for (curr_map_it = curr_objName_cloud_map.begin(); curr_map_it != curr_objName_cloud_map.end(); curr_map_it++) {
        const std::string &curr_obj_name = curr_map_it->first;
        if (ref_objName_cloud_map.find(curr_obj_name) == ref_objName_cloud_map.end()) { //curr object new
            for (size_t i = 0; i < curr_map_it->second->points.size(); i++) {
                curr_map_it->second->points[i].label = ObjectClass::NEW;
            }
            curr_gt_objects.push_back(GTObject(curr_map_it->first, curr_map_it->second, ObjectClass::NEW));
            gt_measurements.nr_novel_obj++;
            *curr_GT_cloud += *(curr_map_it->second);
        }
    }

    gt_measurements.nr_det_obj = gt_measurements.nr_novel_obj + gt_measurements.nr_removed_obj + gt_measurements.nr_static_obj + gt_measurements.nr_moved_obj;

    GTLabeledClouds gt_labeled_clouds;
    gt_labeled_clouds.ref_GT_cloud = ref_GT_cloud;
    gt_labeled_clouds.curr_GT_cloud = curr_GT_cloud;

    //we downsample the GT clouds accordingly to the leaf size used to create the results
    gt_labeled_clouds.ref_GT_cloud = downsampleCloud(gt_labeled_clouds.ref_GT_cloud, ds_voxel_size);
    gt_labeled_clouds.curr_GT_cloud = downsampleCloud(gt_labeled_clouds.curr_GT_cloud, ds_voxel_size);
    for (GTObject& o : ref_gt_objects)
        o.object_cloud = downsampleCloud(o.object_cloud, ds_voxel_size);
    for (GTObject& o : curr_gt_objects)
        o.object_cloud = downsampleCloud(o.object_cloud, ds_voxel_size);


    gt_cloud_numbers.clouds = gt_labeled_clouds;
    gt_cloud_numbers.m = gt_measurements;
    gt_cloud_numbers.ref_objects = ref_gt_objects;
    gt_cloud_numbers.curr_objects = curr_gt_objects;

    return gt_cloud_numbers;

}


std::vector<std::string> checkRemovedObjects(pcl::PointCloud<PointLabel>::Ptr removed_obj_cloud, const GTCloudsAndNumbers gt_cloud_numbers,
                                             std::map<std::string, int> &tp_class_object_count) {
    PointLabel nan_point;
    nan_point.x = nan_point.y = nan_point.z = std::numeric_limits<float>::quiet_NaN();

    /// create tree for radius search
    pcl::KdTreeFLANN<PointLabel> tree;
    std::vector<int> nn_indices;
    std::vector<float> nn_distances;

    /// Check REMOVED objects
    std::vector<std::string> ref_matched_removed;
    if (!removed_obj_cloud->empty()) {
        tree.setInputCloud(removed_obj_cloud);

        for (GTObject gt_obj : gt_cloud_numbers.ref_objects) {
            //only objects that were counted as detected are checked if they are correctly categorized
            if (gt_obj.is_detected) {
                bool is_obj_matched = false;
                if (gt_obj.class_label == ObjectClass::REMOVED) {
                    for (size_t p = 0; p < gt_obj.object_cloud->size(); ++p) {
                        const PointLabel &p_object = gt_obj.object_cloud->points[p];
                        if (!pcl::isFinite(p_object))
                            continue;
                        if (tree.radiusSearch(p_object, search_radius, nn_indices, nn_distances) > 0){
                            if (!is_obj_matched)
                                ref_matched_removed.push_back(gt_obj.name);
                            is_obj_matched = true;
                            /// remove overlapping points (=correctly classified points) from the result cloud
                            for (size_t k = 0; k < nn_indices.size(); k++) {
                                removed_obj_cloud->points[nn_indices[k]] = nan_point;
                            }
                        }
                    }
                }
                if (is_obj_matched) {
                    is_obj_matched = false;
                    std::map<std::string, int>::iterator it = tp_class_object_count.find(gt_obj.name);
                    if (it == tp_class_object_count.end())
                        tp_class_object_count[gt_obj.name] = 1;
                    else
                        tp_class_object_count[gt_obj.name] += 1;
                }
            }
        }
    }

    return ref_matched_removed;
}

std::vector<std::string> checkNovelObjects(pcl::PointCloud<PointLabel>::Ptr novel_obj_cloud, const GTCloudsAndNumbers gt_cloud_numbers,
                                           std::map<std::string, int> &tp_class_object_count) {
    PointLabel nan_point;
    nan_point.x = nan_point.y = nan_point.z = std::numeric_limits<float>::quiet_NaN();

    /// create tree for radius search
    pcl::KdTreeFLANN<PointLabel> tree;
    std::vector<int> nn_indices;
    std::vector<float> nn_distances;

    /// Check REMOVED objects
    std::vector<std::string> curr_matched_novel;
    if (!novel_obj_cloud->empty()) {
        tree.setInputCloud(novel_obj_cloud);
        for (GTObject gt_obj : gt_cloud_numbers.curr_objects) {
            if (gt_obj.is_detected) {
                bool is_obj_matched = false;
                if (gt_obj.class_label == ObjectClass::NEW) {
                    for (size_t p = 0; p < gt_obj.object_cloud->size(); ++p) {
                        const PointLabel &p_object = gt_obj.object_cloud->points[p];
                        if (!pcl::isFinite(p_object))
                            continue;
                        if (tree.radiusSearch(p_object, search_radius, nn_indices, nn_distances) > 0){
                            if (!is_obj_matched)
                                curr_matched_novel.push_back(gt_obj.name);
                            is_obj_matched = true;
                            /// remove overlapping points (=correctly classified points) from the result cloud
                            for (size_t k = 0; k < nn_indices.size(); k++) {
                                novel_obj_cloud->points[nn_indices[k]] = nan_point;
                            }
                        }
                    }
                }
                if (is_obj_matched) {
                    is_obj_matched = false;
                    std::map<std::string, int>::iterator it = tp_class_object_count.find(gt_obj.name);
                    if (it == tp_class_object_count.end())
                        tp_class_object_count[gt_obj.name] = 1;
                    else
                        tp_class_object_count[gt_obj.name] += 1;
                }
            }
        }

        return curr_matched_novel;
    }
}

std::tuple<std::vector<std::string>, std::vector<std::string>> checkDisplacedObjects(pcl::PointCloud<PointLabel>::Ptr c_moved_obj_cloud, pcl::PointCloud<PointLabel>::Ptr r_moved_obj_cloud,
                                                                                     pcl::PointCloud<PointLabel>::Ptr c_static_obj_cloud, pcl::PointCloud<PointLabel>::Ptr r_static_obj_cloud,
                                                                                     const GTCloudsAndNumbers gt_cloud_numbers, std::map<std::string, int> &tp_class_object_count) {
    PointLabel nan_point;
    nan_point.x = nan_point.y = nan_point.z = std::numeric_limits<float>::quiet_NaN();

    /// create tree for radius search
    std::vector<int> nn_indices;
    std::vector<float> nn_distances;
    pcl::KdTreeFLANN<PointLabel> c_tree;
    pcl::KdTreeFLANN<PointLabel> r_tree;

    pcl::PointCloud<PointLabel>::Ptr c_cloud, r_cloud;

    const std::vector<GTObject> &gt_ref_objects = gt_cloud_numbers.ref_objects;

    std::vector<std::string> matched_moved;
    std::vector<std::string> matched_static;

    for (GTObject gt_obj : gt_cloud_numbers.curr_objects) {
        if (gt_obj.is_detected) {
            std::vector<int> c_overlapping_indices, r_overlapping_indices;
            bool is_static=true;
            if (gt_obj.class_label ==ObjectClass::REMOVED || gt_obj.class_label == ObjectClass::NEW) {
                continue;
            }
            if (categoryTransformer(gt_obj.class_label) == ObjectClass::MOVED) { //MOVED
                if (c_moved_obj_cloud->empty() || r_moved_obj_cloud->empty())
                    continue;
                c_cloud = c_moved_obj_cloud;
                r_cloud = r_moved_obj_cloud;
                is_static = false;
            } else if (categoryTransformer(gt_obj.class_label) == ObjectClass::STATIC) { //STATIC
                if (c_static_obj_cloud->empty() || r_static_obj_cloud->empty())
                    continue;
                c_cloud = c_static_obj_cloud;
                r_cloud = r_static_obj_cloud;
                is_static = true;
            } else {
                std::cerr << "Something is wrong. This label should not exist!" << std::endl;
            }
            c_tree.setInputCloud(c_cloud);
            r_tree.setInputCloud(r_cloud);

            /// find the corresponding ref object
            std::vector<GTObject>::const_iterator ref_it = find_if(gt_ref_objects.begin(), gt_ref_objects.end(),
                                                                   [gt_obj](const GTObject ref_obj){return gt_obj.class_label == ref_obj.class_label;});
            if (ref_it->is_detected) {
                //get overlapping points from current scene
                for (size_t p = 0; p < gt_obj.object_cloud->size(); ++p) {
                    const PointLabel &p_object = gt_obj.object_cloud->points[p];
                    if (!pcl::isFinite(p_object))
                        continue;
                    if (c_tree.radiusSearch(p_object, search_radius, nn_indices, nn_distances) > 0){
                        c_overlapping_indices.insert(c_overlapping_indices.end(), nn_indices.begin(), nn_indices.end());
                    }
                }
                //get overlapping points from reference scene
                for (size_t p = 0; p < ref_it->object_cloud->size(); ++p) {
                    const PointLabel &p_object = ref_it->object_cloud->points[p];
                    if (!pcl::isFinite(p_object))
                        continue;
                    if (r_tree.radiusSearch(p_object, search_radius, nn_indices, nn_distances) > 0){
                        r_overlapping_indices.insert(r_overlapping_indices.end(), nn_indices.begin(), nn_indices.end());
                    }
                }

                //sort and remove double indices
                std::sort(c_overlapping_indices.begin(), c_overlapping_indices.end());
                c_overlapping_indices.erase(unique(c_overlapping_indices.begin(), c_overlapping_indices.end()), c_overlapping_indices.end());
                std::sort(r_overlapping_indices.begin(), r_overlapping_indices.end());
                r_overlapping_indices.erase(unique(r_overlapping_indices.begin(), r_overlapping_indices.end()), r_overlapping_indices.end());

                if (r_overlapping_indices.size() == 0 || c_overlapping_indices.size() == 0)
                    continue;

                //find the label that is most often occuring in the overlap
                std::vector<int> r_label_vec, c_label_vec;
                for (auto p : r_overlapping_indices)
                    r_label_vec.push_back(r_cloud->points[p].label);
                for (auto p : c_overlapping_indices)
                    c_label_vec.push_back(c_cloud->points[p].label);
                int r_most_label = getMostFrequentNumber(r_label_vec);
                int c_most_label = getMostFrequentNumber(c_label_vec);

                /// correct match
                if (r_most_label == c_most_label) {
                    if (is_static) {
                        matched_static.push_back(gt_obj.name);
                    }
                    else {
                        matched_moved.push_back(gt_obj.name);
                    }
                    std::map<std::string, int>::iterator it = tp_class_object_count.find(gt_obj.name);
                    if (it == tp_class_object_count.end())
                        tp_class_object_count[gt_obj.name] = 2;
                    else
                        tp_class_object_count[gt_obj.name] += 2;

                    //update the clouds and remove matched indices
                    pcl::PointIndices::Ptr c_ind(new pcl::PointIndices());
                    for (auto p : r_overlapping_indices) {
                        if (r_cloud->points[p].label == r_most_label)
                            c_ind->indices.push_back(p);
                    }
                    pcl::ExtractIndices<PointLabel> extract;
                    extract.setInputCloud (r_cloud);
                    extract.setIndices(c_ind);
                    extract.setKeepOrganized(true);
                    extract.setNegative (true);
                    extract.filter(*r_cloud);

                    c_ind.reset(new pcl::PointIndices());
                    for (auto p : c_overlapping_indices) {
                        if (c_cloud->points[p].label == r_most_label)
                            c_ind->indices.push_back(p);
                    }
                    extract.setInputCloud (c_cloud);
                    extract.setIndices(c_ind);
                    extract.filter(*c_cloud);
                }
            }
        }
    }
    std::tuple<std::vector<std::string>, std::vector<std::string>> matched_objects;
    matched_objects = std::make_tuple(matched_static, matched_moved);
    return matched_objects;
}

int categoryTransformer(int cat_label) {
    if (cat_label == 2)
        return ObjectClass::REMOVED;

    if (cat_label == 3)
        return ObjectClass::NEW;

    if (cat_label == 1 || (cat_label >= 1000 && cat_label <= 1950))  //MOVED
        return ObjectClass::MOVED;

    if (cat_label == 0 || (cat_label >= 20 && cat_label <= 950))  //STATIC
        return ObjectClass::STATIC;
}

void computeConfusionMat(GTCloudsAndNumbers gt_objects, pcl::PointCloud<PointLabel>::Ptr ref_result_cloud,
                         pcl::PointCloud<PointLabel>::Ptr curr_result_cloud, std::vector<std::vector<int>> &conf_mat) {
    std::vector<int> indices;
    pcl::PointCloud<PointLabel>::Ptr ref_cloud_wo_nans(new  pcl::PointCloud<PointLabel>);
    pcl::removeNaNFromPointCloud(*ref_result_cloud, *ref_cloud_wo_nans, indices);
    pcl::PointCloud<PointLabel>::Ptr curr_cloud_wo_nans(new  pcl::PointCloud<PointLabel>);
    pcl::removeNaNFromPointCloud(*curr_result_cloud, *curr_cloud_wo_nans, indices);

    pcl::octree::OctreePointCloudSearch<PointLabel> r_octree (0.2);
    r_octree.setInputCloud (ref_cloud_wo_nans);
    r_octree.addPointsFromInputCloud ();
    pcl::octree::OctreePointCloudSearch<PointLabel> c_octree (0.2);
    c_octree.setInputCloud (curr_cloud_wo_nans);
    c_octree.addPointsFromInputCloud ();

    for (GTObject obj : gt_objects.ref_objects) {
        int obj_cat = categoryTransformer(obj.class_label);
        std::vector<int> r_overlapping_ind;
        std::vector<int> nn_indices;
        std::vector<float> nn_distances;

        for (size_t p = 0; p < obj.object_cloud->size(); ++p) {
            const PointLabel &p_object = obj.object_cloud->points[p];
            if (!pcl::isFinite(p_object))
                continue;
            if (r_octree.radiusSearch(p_object, search_radius, nn_indices, nn_distances) > 0){
                r_overlapping_ind.insert(r_overlapping_ind.end(), nn_indices.begin(), nn_indices.end());
            }
        }
        std::sort(r_overlapping_ind.begin(), r_overlapping_ind.end());
        r_overlapping_ind.erase(std::unique( r_overlapping_ind.begin(), r_overlapping_ind.end() ), r_overlapping_ind.end() );

        if (r_overlapping_ind.size() == 0)
            continue;
        std::vector<int> overlapping_labels;
        for (size_t i = 0; i < r_overlapping_ind.size(); i++) {
            int label = categoryTransformer(ref_cloud_wo_nans->points.at(r_overlapping_ind[i]).label);
            overlapping_labels.push_back(label);
        }

        if (obj_cat == ObjectClass::REMOVED) {
            int category = getMostFrequentNumber(overlapping_labels);
            conf_mat[obj_cat][category] += 1;
        }
        else {
            /// find the corresponding curr object
            std::vector<GTObject>::const_iterator curr_it = find_if(gt_objects.curr_objects.begin(), gt_objects.curr_objects.end(),
                                                                    [obj](const GTObject curr_obj){return obj.class_label == curr_obj.class_label;});

            //get overlapping points from current scene
            std::vector<int> c_overlapping_ind;
            for (size_t p = 0; p < curr_it->object_cloud->size(); ++p) {
                const PointLabel &p_object = curr_it->object_cloud->points[p];
                if (!pcl::isFinite(p_object))
                    continue;
                if (c_octree.radiusSearch(p_object, search_radius, nn_indices, nn_distances) > 0){
                    c_overlapping_ind.insert(c_overlapping_ind.end(), nn_indices.begin(), nn_indices.end());
                }
            }

            //sort and remove double indices
            std::sort(c_overlapping_ind.begin(), c_overlapping_ind.end());
            c_overlapping_ind.erase(unique(c_overlapping_ind.begin(), c_overlapping_ind.end()), c_overlapping_ind.end());

            if (c_overlapping_ind.size() == 0) {
                int category = getMostFrequentNumber(overlapping_labels);
                conf_mat[obj_cat][category] += 1;
            }
            else {

                //find the label that is most often occuring in the overlap
                std::vector<int> r_label_vec, c_label_vec;
                for (auto p : r_overlapping_ind)
                    r_label_vec.push_back(ref_cloud_wo_nans->points[p].label);
                for (auto p : c_overlapping_ind)
                    c_label_vec.push_back(curr_cloud_wo_nans->points[p].label);
                int r_most_label = getMostFrequentNumber(r_label_vec);
                int c_most_label = getMostFrequentNumber(c_label_vec);

                int r_category = categoryTransformer(r_most_label);
                int c_category = categoryTransformer(c_most_label);

                if (r_category == c_category) {
                    /// correct match
                    if (r_most_label == c_most_label) {
                        conf_mat[obj_cat][r_category] += 1;
                        conf_mat[obj_cat][c_category] += 1;
                    } else {
                        conf_mat[obj_cat][4] += 2;
                    }
                } else {
                    conf_mat[obj_cat][r_category] += 1;
                    conf_mat[obj_cat][c_category] += 1;
                }
            }
        }
    }

    for (GTObject obj : gt_objects.curr_objects) {
        int obj_cat = categoryTransformer(obj.class_label);
        if (obj_cat == ObjectClass::NEW) {
            std::vector<int> c_overlapping_ind;
            std::vector<int> nn_indices;
            std::vector<float> nn_distances;

            for (size_t p = 0; p < obj.object_cloud->size(); ++p) {
                const PointLabel &p_object = obj.object_cloud->points[p];
                if (!pcl::isFinite(p_object))
                    continue;
                if (c_octree.radiusSearch(p_object, 0.00001, nn_indices, nn_distances) > 0){
                    c_overlapping_ind.insert(c_overlapping_ind.end(), nn_indices.begin(), nn_indices.end());
                }
            }
            std::sort(c_overlapping_ind.begin(), c_overlapping_ind.end());
            c_overlapping_ind.erase(std::unique( c_overlapping_ind.begin(), c_overlapping_ind.end() ), c_overlapping_ind.end() );

            if (c_overlapping_ind.size() == 0)
                continue;
            std::vector<int> overlapping_labels;
            for (size_t i = 0; i < c_overlapping_ind.size(); i++) {
                int label = categoryTransformer(curr_cloud_wo_nans->points.at(c_overlapping_ind[i]).label);
                overlapping_labels.push_back(label);
            }

            int category = getMostFrequentNumber(overlapping_labels);
            conf_mat[obj_cat][category] += 1;
        }
    }
}

/*void computeConfusionMat(std::vector<GTObject> gt_objects, pcl::PointCloud<PointLabel>::Ptr result_cloud, std::vector<std::vector<int>> &conf_mat) {
    std::vector<int> indices;
    pcl::PointCloud<PointLabel>::Ptr cloud_wo_nans(new  pcl::PointCloud<PointLabel>);
    pcl::removeNaNFromPointCloud(*result_cloud, *cloud_wo_nans, indices);

    pcl::octree::OctreePointCloudSearch<PointLabel> octree (0.2);
    octree.setInputCloud (cloud_wo_nans);
    octree.addPointsFromInputCloud ();

    for (GTObject obj : gt_objects) {
        std::vector<int> overlapping_ind;
        std::vector<int> nn_indices;
        std::vector<float> nn_distances;

        for (size_t p = 0; p < obj.object_cloud->size(); ++p) {
            const PointLabel &p_object = obj.object_cloud->points[p];
            if (!pcl::isFinite(p_object))
                continue;
            if (octree.radiusSearch(p_object, 0.00001, nn_indices, nn_distances) > 0){
                overlapping_ind.insert(overlapping_ind.end(), nn_indices.begin(), nn_indices.end());
            }
        }
        std::sort(overlapping_ind.begin(), overlapping_ind.end());
        overlapping_ind.erase(std::unique( overlapping_ind.begin(), overlapping_ind.end() ), overlapping_ind.end() );

        if (overlapping_ind.size() == 0)
            continue;
        std::vector<int> overlapping_labels;
        for (size_t i = 0; i < overlapping_ind.size(); i++) {
            int label = categoryTransformer(cloud_wo_nans->points.at(overlapping_ind[i]).label);
            overlapping_labels.push_back(label);
        }
        int category = getMostFrequentNumber(overlapping_labels);
        conf_mat[categoryTransformer(obj.class_label)][category] += 1;
    }
}*/

//relabel static and moved objects to be in the range 20-950 resp. 1000-1950
void relabelMatches(pcl::PointCloud<PointLabel>::Ptr ref_cloud, pcl::PointCloud<PointLabel>::Ptr curr_cloud, bool isStatic) {
    std::set<int> available_labels;
    for (PointLabel p : ref_cloud->points) {
        available_labels.insert(p.label);
    }

    std::set<int> label_check;
    for (PointLabel p : curr_cloud->points) {
        if (available_labels.find(p.label) == available_labels.end()) {
            std::cout << "Found a label in the current cloud that was not in the ref cloud!" << std::endl;
            exit(-1);
        }
        label_check.insert(p.label);
    }

    if (available_labels.size() != label_check.size()) {
        std::cout << "The number of matched objects in ref and curr is different!" << std::endl;
        exit(-1);
    }

    int new_label;
    if (isStatic)
        new_label = 20;
    else
        new_label = 1000;

    for (int label : available_labels) {
        for (PointLabel &p : ref_cloud->points) {
            if (p.label == label)
                p.label = new_label;
        }
        for (PointLabel &p : curr_cloud->points) {
            if (p.label == label)
                p.label = new_label;
        }
        new_label += 10;
    }
}
