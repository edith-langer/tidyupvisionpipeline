#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <v4r/io/filesystem.h>

#include <glog/logging.h>

#include "ppf_recognition_pipeline.h"

namespace py = pybind11;
namespace po = boost::program_options;
namespace bf = boost::filesystem;

using PointT = pcl::PointXYZRGB;
using Hypotheses = std::vector<std::vector<v4r::ObjectHypothesis::Ptr>>;

class PPFPythonWrapper {
public:

  PPFPythonWrapper(const std::string& cfg_dir, const std::string& models_dir, const std::vector<std::string>& object_models, bool retrain = false);

  // @todo: add ability to set sensor origin and orientation, take Eigen::Quaternionf and Eigen::Vector4f args
  Hypotheses estimate(py::array_t<double> points_array, const std::vector<std::string>& obj_models_to_search = {});

private:
  v4r::PPFRecognitionPipelineParameter param_;
  v4r::NormalEstimatorParameter normal_estimator_param_;
  v4r::PPFRecognitionPipeline<PointT>::Ptr pipeline_;
  v4r::Source<PointT>::Ptr models_db_;
  std::string models_dir_;
  std::string cfg_dir_;
};

PPFPythonWrapper::PPFPythonWrapper(const std::string& cfg_dir, const std::string& models_dir, const std::vector<std::string>& object_models, bool retrain) :
models_dir_{models_dir},
cfg_dir_{cfg_dir} {

  po::options_description desc("Options for ppf recognition pipeline");
  po::variables_map vm;

  param_.init(desc, "ppf_pipeline");
  normal_estimator_param_.init(desc, "normal_estimator");

  auto config_file = bf::path(cfg_dir_) / "v4r_object_reco_config_ppf.ini";
  if (v4r::io::existsFile(config_file)) {
    std::ifstream f(config_file.string());
    po::parsed_options config_parsed = po::parse_config_file(f, desc, true);  // true -> allow unregistered options
    po::store(config_parsed, vm);
    f.close();
  } else {
    LOG(ERROR) << config_file.string() << " does not exist! Usage: " << desc;
  }

  try {
    po::notify(vm);
  } catch (const po::error &e) {
    std::cerr << "Error: " << e.what() << std::endl << std::endl << desc << std::endl;
  }

  models_db_.reset(new v4r::Source<PointT>(v4r::SourceParameter()));
  models_db_->init(bf::path(models_dir_), object_models);

  pipeline_.reset(new v4r::PPFRecognitionPipeline<PointT>(param_));
  pipeline_->setModelDatabase(models_db_);

  pipeline_->initialize(bf::path(models_dir), retrain, object_models);
  models_db_->cleanUpTrainingData(true);
}

  Hypotheses PPFPythonWrapper::estimate(py::array_t<double> points_array, const std::vector<std::string>& obj_models_to_search) {

  auto points_info = points_array.request();
  if (points_info.ndim != 2) {
    throw std::runtime_error("Expect array of shape (num_points, XYZRGB).");
  }
  if (points_info.shape[1] != 6) {
    throw std::runtime_error("Expect array of shape (num_points, XYZRGB).");
  }

  auto num_points = points_info.shape[0];
  auto points = static_cast<double*>(points_info.ptr);

  // parse input buffer to point cloud
  pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);

  for (decltype(num_points) i = 0; i < num_points; i++) {
    auto index = i*6;
    PointT point;
    point.x = static_cast<float>(points[index]);
    point.y = static_cast<float>(points[index+1]);
    point.z = static_cast<float>(points[index+2]);
    point.r = static_cast<uint8_t>(points[index+3] * 255);
    point.g = static_cast<uint8_t>(points[index+4] * 255);
    point.b = static_cast<uint8_t>(points[index+5] * 255);
    cloud->push_back(point);
  }

  // // set point cloud sensor origin / orientation
  // input_cloud->sensor_orientation_ = ...
  // input_cloud->sensor_origin_ = ...

  // check if object models to search for exist in model database
  for (const auto &id : obj_models_to_search) {
    if (!models_db_->getModelById("", id))
      LOG(ERROR) << "Requested object model \"" << id << "\" is not found in model database " << models_db_;
  }

  std::vector<std::string> models;
  for (const auto &m : models_db_->getModels()) {
    if (obj_models_to_search.empty() || std::find(obj_models_to_search.begin(), obj_models_to_search.end(),
                                                      m->id_) != obj_models_to_search.end()) {
      models.push_back(m->id_);
    }
  }

  if (models.empty()) {
    LOG(ERROR) << "No valid objects to search for specified!";
    return {};
  } else {
    std::stringstream info_txt;
    info_txt << "Searching for following object model(s): ";
    for (const auto &m : models)
      info_txt << m << ", ";
    LOG(INFO) << info_txt.str();
  }

  // compute normals
  pcl::PointCloud<pcl::Normal>::Ptr normals;
  if (pipeline_->needNormals()) {
    normals = v4r::computeNormals<PointT>(cloud, normal_estimator_param_);
    pipeline_->setSceneNormals(normals);
  }

  // LOG(INFO) << "pointcloud num points " << cloud->points.size();
  // LOG(INFO) << "downsampling resoution : " << param_.downsampling_resolution_;

  pipeline_->setInputCloud(cloud);
  // if (transform_to_world)
    // pipeline_->setTransformToWorld(transform_to_world.get());
  pipeline_->recognize(models);
  auto generated_object_hypotheses = pipeline_->getObjectHypothesis();

  // return hypotheses in a way that makes them convenient to use from python
  std::vector<std::vector<v4r::ObjectHypothesis::Ptr>> hypotheses;
  for (auto& ohg : generated_object_hypotheses) {
    hypotheses.push_back(ohg.ohs_);
  }

  return hypotheses;
}

PYBIND11_MODULE(pyppf, m) {
  py::class_<v4r::ObjectHypothesis, v4r::ObjectHypothesis::Ptr>(m, "ObjectHypothesis")
    .def(py::init<>())
    // .def_readwrite("class_id", &v4r::ObjectHypothesis::class_id_)    // not actually used by ppf
    .def_readwrite("model_id", &v4r::ObjectHypothesis::model_id_)
    .def_readwrite("transform", &v4r::ObjectHypothesis::transform_)
    // .def_readwrite("pose_refinement", &v4r::ObjectHypothesis::pose_refinement_)  // not actually used by ppf
    .def_readwrite("confidence", &v4r::ObjectHypothesis::confidence_wo_hv_);

  py::class_<PPFPythonWrapper>(m, "PPF")
    .def(py::init<const std::string&, const std::string&, const std::vector<std::string>&>(),
      py::arg("cfg_dir"), py::arg("models_dir"),
      py::arg("object_models") = std::vector<std::string>())
    .def("estimate", &PPFPythonWrapper::estimate,
      py::arg("points_array"),
      py::arg("obj_models_to_search") = std::vector<std::string>());
}
