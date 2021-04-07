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

#include <PPFRecognizer.h>


typedef pcl::PointXYZRGB PointRGB;
typedef pcl::PointXYZRGBNormal PointNormal;
typedef pcl::PointXYZRGBL PointLabel;

//boost graph defs
struct VertexProperty {
    std::string name;
    VertexProperty(std::string name_) {name=name_;}
    VertexProperty(){}
};

//we use that to store the transformation from object to model as edge property
struct transformation_t {typedef boost::edge_property_tag kind;};
typedef boost::property<transformation_t, Eigen::Matrix4f > Transformnation_e_prop;

typedef boost::property< boost::edge_weight_t, float, Transformnation_e_prop > EdgeProperty;
typedef boost::adjacency_list< boost::vecS, boost::vecS, boost::undirectedS, VertexProperty, EdgeProperty > my_graph;

using vertex_t = boost::graph_traits<my_graph>::vertex_descriptor;
using edge_t   = boost::graph_traits<my_graph>::edge_descriptor;

//result defs
struct Match {
    int model_id;
    int object_id;
    float confidence;
    Eigen::Matrix4f transform;
    Match(int m_id, int o_id, float conf, Eigen::Matrix4f t) {model_id = m_id; object_id = o_id; confidence = conf; transform = t;}
};


struct ObjectHypothesesStruct {
    int object_id;
    pcl::PointCloud<PointNormal>::Ptr object_cloud;
    pcl::PointCloud<PointNormal>::Ptr obj_pts_not_explained_cloud;
    std::vector<v4r::ObjectHypothesesGroup> hypotheses; //return value of the recognize-method: each element contains the hypothesis for a model
};

class ObjectMatching
{
public:
    ObjectMatching(std::vector<DetectedObject> model_vec, std::vector<DetectedObject> object_vec,
                   std::string model_path, std::string cfg_path);

    std::vector<Match> compute();

private:
    std::vector<DetectedObject> model_vec_;
    std::vector<DetectedObject> object_vec_;
    std::string model_path_;
    std::string cfg_path_;

    std::tuple<float, float> computeModelFitness(pcl::PointCloud<PointNormal>::Ptr object, pcl::PointCloud<PointNormal>::Ptr model,
                                                 v4r::apps::PPFRecognizerParameter param);
    void saveCloudResults(pcl::PointCloud<PointNormal>::Ptr object_cloud, pcl::PointCloud<PointNormal>::Ptr model_aligned,
                          pcl::PointCloud<PointNormal>::Ptr model_aligned_refined, std::string path);
    bool isModelBelowPlane(pcl::PointCloud<PointNormal>::Ptr model, pcl::PointCloud<PointNormal>::Ptr plane_cloud);
    std::vector<Match> weightedGraphMatching(std::vector<ObjectHypothesesStruct> global_hypotheses);
};

#endif // OBJECT_MATCHING_H
