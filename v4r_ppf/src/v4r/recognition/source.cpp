#include <glog/logging.h>
#include <pcl/point_types.h>
#include <v4r/io/filesystem.h>
#include <v4r/recognition/source.h>
#include <boost/algorithm/string.hpp>

namespace v4r {

template <typename PointT>
void Source<PointT>::init(const bf::path &model_database_path,
                          const std::vector<std::string> &object_instances_to_load) {
  std::vector<std::string> categories;
  if (param_.has_categories_)
    categories = io::getFoldersInDirectory(model_database_path);
  else
    categories.push_back("");

  for (const std::string &cat : categories) {
    const bf::path class_path = model_database_path / cat;
    const std::vector<std::string> instance_names = io::getFoldersInDirectory(class_path);

    LOG(INFO) << "Loading " << instance_names.size() << " object models from folder " << class_path.string() << ". ";

    if (!object_instances_to_load.empty()) {
      // check if all given objects are present
      for (const std::string &id : object_instances_to_load) {
        CHECK(std::find(instance_names.begin(), instance_names.end(), id) != instance_names.end())
            << "Given object model to load (" << id << ") is not present in the object model directory!";
      }
    }

    for (const std::string &instance_name : instance_names) {
      if (!object_instances_to_load.empty() &&
          std::find(object_instances_to_load.begin(), object_instances_to_load.end(), instance_name) ==
              object_instances_to_load.end()) {
        LOG(INFO) << "Skipping object " << instance_name << " because it is not in the list of objects to load.";
        continue;
      }

      typename Model<PointT>::Ptr obj(new Model<PointT>);
      obj->id_ = instance_name;
      obj->class_ = cat;

      const bf::path object_dir = class_path / instance_name / param_.view_folder_name_;
      const std::string view_pattern = ".*" + param_.view_prefix_ + ".*.pcd";
      std::vector<std::string> training_view_filenames = io::getFilesInDirectory(object_dir, view_pattern, false);

      LOG(INFO) << " ** loading model (class: " << cat << ", instance: " << instance_name << ") with "
                << training_view_filenames.size() << " views. ";

      for (size_t v_id = 0; v_id < training_view_filenames.size(); v_id++) {
        typename TrainingView<PointT>::Ptr v(new TrainingView<PointT>);
        v->filename_ = object_dir / training_view_filenames[v_id];

        std::string pose_filename = v->filename_.string();
        boost::replace_last(pose_filename, param_.view_prefix_, param_.pose_prefix_);
        boost::replace_last(pose_filename, ".pcd", ".txt");
        v->pose_filename_ = pose_filename;

        std::string indices_filename = v->filename_.string();
        boost::replace_last(indices_filename, param_.view_prefix_, param_.indices_prefix_);
        boost::replace_last(indices_filename, ".pcd", ".txt");
        v->indices_filename_ = indices_filename;

        obj->addTrainingView(v);
      }

      if (!param_.has_categories_) {
        bf::path model3D_path = class_path / instance_name / param_.name_3D_model_;
        obj->initialize(model3D_path);
      }
      obj->properties_ = ModelProperties(class_path / instance_name / param_.metadata_name_);
      addModel(obj);
    }
  }
}

template <typename PointT>
typename Model<PointT>::ConstPtr Source<PointT>::getModelById(const std::string &class_id,
                                                              const std::string &instance_id) const {
  for (size_t i = 0; i < models_.size(); i++) {
    if (models_[i]->id_.compare(instance_id) == 0 && models_[i]->class_.compare(class_id) == 0) {
      return models_[i];
    }
  }
  LOG(ERROR) << "Model with class: " << class_id << " and instance: " << instance_id << " not found";
  return nullptr;
}

template class Source<pcl::PointXYZ>;
template class Source<pcl::PointXYZRGB>;
}  // namespace v4r
