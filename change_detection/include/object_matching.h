#ifndef OBJECT_MATCHING_H
#define OBJECT_MATCHING_H

#include <iostream>
#include <regex>
#include <chrono>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <pcl/registration/warp_point_rigid_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_lm.h>
#include <pcl/console/parse.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/segment_differences.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/concave_hull.h>

#include <boost/program_options.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost_graph/maximum_weighted_matching.hpp>

#include <glog/logging.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <v4r/recognition/object_hypothesis.h>
#include <v4r/common/color_comparison.h>
#include <v4r/geometry/normals.h>

#include "plane_object_extraction.h"
#include "warp_point_rigid_4d.h"
#include "scene_differencing_points.h"
#include "color_histogram.h"
#include "detected_object.h"
#include "mathhelpers.h"

#include <PPFRecognizer.h>

#include "settings.h"


typedef pcl::PointXYZRGB PointRGB;
typedef pcl::PointXYZRGBNormal PointNormal;
typedef pcl::PointXYZRGBL PointLabel;

static const float point_dist_diff = 0.005f;

//boost graph defs
struct VertexProperty {
    std::string name;
    VertexProperty(std::string name_) {name=name_;}
    VertexProperty(){}
};

struct HypothesesStruct {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    HypothesesStruct() {}
    HypothesesStruct(int id, pcl::PointCloud<PointNormal>::Ptr cloud, Eigen::Matrix4f transf, FitnessScoreStruct fitness_score) :
        model_id(id), transform(transf), fitness(fitness_score) {}
    int model_id;
    Eigen::Matrix<float,4,4,Eigen::DontAlign> transform;  ///< 4x4 homogenous transformation to project model into camera coordinate system.
    FitnessScoreStruct fitness;
};

struct ObjectHypothesesStruct {
    int object_id;
    pcl::PointCloud<PointNormal>::ConstPtr object_cloud;
    std::map<int, HypothesesStruct> model_hyp; //store for each model hypotheses the fitness
};

//we use that to store the transformation from object to model as edge property
struct hypo_t {typedef boost::edge_property_tag kind;};
typedef boost::property<hypo_t, HypothesesStruct > hypo_e_prop;

typedef boost::property< boost::edge_weight_t, float, hypo_e_prop > EdgeProperty;
typedef boost::adjacency_list< boost::vecS, boost::vecS, boost::undirectedS, VertexProperty, EdgeProperty > my_graph;

using vertex_t = boost::graph_traits<my_graph>::vertex_descriptor;
using edge_t   = boost::graph_traits<my_graph>::edge_descriptor;





class ObjectMatching
{
public:
    ObjectMatching(std::vector<DetectedObject> model_vec, std::vector<DetectedObject> object_vec,
                   std::string model_path, std::string cfg_path, std::string obj_match_dir="");

    std::vector<Match> compute(std::vector<DetectedObject> &ref_result, std::vector<DetectedObject> &curr_result);
    static FitnessScoreStruct computeModelFitness(pcl::PointCloud<PointNormal>::ConstPtr object, pcl::PointCloud<PointNormal>::ConstPtr model,
                                                        v4r::apps::PPFRecognizerParameter param);
    static float estimateDistance(const pcl::PointCloud<PointNormal>::ConstPtr object_cloud, const pcl::PointCloud<PointNormal>::ConstPtr model_cloud, const Eigen::Matrix4f transform);
    static std::vector<pcl::PointIndices> clusterOutliersBySize(const pcl::PointCloud<PointNormal>::ConstPtr cloud, std::vector<int> &removed_ind, float cluster_thr,
                                                                int min_cluster_size=15, int max_cluster_size=std::numeric_limits<int>::max());
    static bool isObjectPlanar(pcl::PointCloud<PointNormal>::ConstPtr object, float plane_dist_thr, float plane_acc_thr);
    void matchedPartGrowing(pcl::PointCloud<PointNormal>::ConstPtr obj_cloud, pcl::PointCloud<PointNormal>::Ptr matched_part,
                                                                         pcl::PointCloud<PointNormal>::Ptr remaining_part, std::vector<int> good_pt_ids);


private:
    std::vector<DetectedObject> model_vec_;
    std::vector<DetectedObject> object_vec_;
    std::string model_path_;
    std::string cfg_path_;
    std::string cloud_matches_dir_;

    boost::shared_ptr<v4r::apps::PPFRecognizer<pcl::PointXYZRGB> > rec_;

    void saveCloudResults(pcl::PointCloud<PointNormal>::ConstPtr object_cloud, pcl::PointCloud<PointNormal>::ConstPtr model_aligned, std::string path);
    bool isBelowPlane(pcl::PointCloud<PointNormal>::ConstPtr model, pcl::PointCloud<PointNormal>::ConstPtr plane_cloud);
    std::vector<Match> weightedGraphMatching(std::vector<ObjectHypothesesStruct> global_hypotheses,
                                             std::function<float(FitnessScoreStruct)> computeFitness, double fitness_thr);
    std::vector<v4r::ObjectHypothesesGroup> callRecognizer(DetectedObject &obj);
    std::pair<HypothesesStruct, bool> filterRecoHypothesis(DetectedObject obj, std::vector<v4r::ObjectHypothesis::Ptr> hg);
    std::vector<ObjectHypothesesStruct> createHypotheses();
    float computeMeanPointDistance(pcl::PointCloud<PointNormal>::ConstPtr ref_object, pcl::PointCloud<PointNormal>::ConstPtr curr_obj);

};

#endif // OBJECT_MATCHING_H
