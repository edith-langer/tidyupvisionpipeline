#include <glog/logging.h>
#include <recognition_pipeline.h>
#include <pcl/impl/instantiate.hpp>

namespace v4r {

template <typename PointT>
RecognitionPipeline<PointT>::StopWatch::~StopWatch() {
  boost::posix_time::ptime end_time = boost::posix_time::microsec_clock::local_time();
  float elapsed_time = static_cast<float>(((end_time - start_time_).total_milliseconds()));
  VLOG(1) << desc_ << " took " << elapsed_time << " ms.";
  elapsed_time_.push_back(std::pair<std::string, float>(desc_, elapsed_time));
}

template <typename PointT>
void RecognitionPipeline<PointT>::deInit() {
  table_plane_set_ = false;
  transform_to_world_set_ = false;
}

template <typename PointT>
void RecognitionPipeline<PointT>::recognize(const std::vector<std::string> &model_ids_to_search,
                                            const boost::optional<Eigen::Matrix4f> &camera_pose) {
  elapsed_time_.clear();
  obj_hypotheses_.clear();
  CHECK(scene_) << "Input scene is not set!";
  CHECK(!model_ids_to_search.empty()) << "Models to search for not specified!";

  if (needNormals())
    CHECK(scene_normals_ && scene_->points.size() == scene_normals_->points.size())
        << "Recognizer needs normals but they are not set!";

  if (camera_pose) {
    transform_to_world_ = camera_pose.get();
    transform_to_world_set_ = true;
  }

  do_recognize(model_ids_to_search);

  deInit();
}

#define PCL_INSTANTIATE_RecognitionPipeline(T) template class RecognitionPipeline<T>;
PCL_INSTANTIATE(RecognitionPipeline, (pcl::PointXYZRGB))
}  // namespace v4r
