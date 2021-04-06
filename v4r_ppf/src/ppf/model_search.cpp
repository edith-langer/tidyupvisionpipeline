/****************************************************************************
**
** Copyright (C) 2019 TU Wien, ACIN, Vision 4 Robotics (V4R) group
** Contact: v4r.acin.tuwien.ac.at
**
** This file is part of V4R
**
** V4R is distributed under dual licenses - GPLv3 or closed source.
**
** GNU General Public License Usage
** V4R is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published
** by the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
**
** V4R is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
** GNU General Public License for more details.
**
** Please review the following information to ensure the GNU General Public
** License requirements will be met: https://www.gnu.org/licenses/gpl-3.0.html.
**
**
** Commercial License Usage
** If GPL is not suitable for your project, you must purchase a commercial
** license to use V4R. Licensees holding valid commercial V4R licenses may
** use this file in accordance with the commercial license agreement
** provided with the Software or, alternatively, in accordance with the
** terms contained in a written agreement between you and TU Wien, ACIN, V4R.
** For licensing terms and conditions please contact office<at>acin.tuwien.ac.at.
**
**
** The copyright holder additionally grants the author(s) of the file the right
** to use, copy, modify, merge, publish, distribute, sublicense, and/or
** sell copies of their contributions without any restrictions.
**
****************************************************************************/

#include <array>
#include <map>

#include <cmph.h>

#include <glog/logging.h>

#include <v4r/common/greedy_local_clustering.h>
#include <ppf/model_search.h>

// Anonymous namespace with local helper functions
namespace {

template <typename T>
T read(FILE* file) {
  T variable;
  if (std::fread(&variable, sizeof(T), 1, file) != 1)
    throw std::runtime_error("failed to read model search from file");
  return variable;
}

template <typename T>
void read(FILE* file, T& variable) {
  if (std::fread(&variable, sizeof(T), 1, file) != 1)
    throw std::runtime_error("failed to read model search from file");
}

template <typename T, typename Allocator>
void read(FILE* file, std::vector<T, Allocator>& vector) {
  vector.resize(read<size_t>(file));
  if (std::fread(vector.data(), sizeof(T), vector.size(), file) != vector.size())
    throw std::runtime_error("failed to read model search from file");
}

template <typename T>
void write(FILE* file, const T& variable) {
  fwrite(&variable, sizeof(T), 1, file);
}

template <typename T, typename Allocator>
void write(FILE* file, const std::vector<T, Allocator>& vector) {
  auto size = vector.size();
  fwrite(&size, sizeof(size), 1, file);
  fwrite(vector.data(), sizeof(T), size, file);
}

void computePPF(const Eigen::Vector3f& p1, const Eigen::Vector3f& n1, const Eigen::Vector3f& p2,
                const Eigen::Vector3f& n2, float f[4]) {
  Eigen::Vector3f delta = p2 - p1;
  f[3] = delta.norm();
  delta /= f[3];
  f[0] = std::acos(n1.dot(delta));
  f[1] = std::acos(n2.dot(delta));
  f[2] = std::acos(n1.dot(n2));
}

void computeCPPF(const Eigen::Vector3f& p1, const Eigen::Vector3f& n1, const Eigen::Vector3f& p2,
                const Eigen::Vector3f& n2, const Eigen::Vector3f& c1, const Eigen::Vector3f& c2, float f[10]) {
  computePPF(p1, n1, p2, n2, f);

  //Color should be in HSV space!
  f[4] = c1[0] / 360.0; //normalise to [0-1]
  f[5] = c1[1];
  f[6] = c1[2];

  f[7]  = c2[0] / 360.0; //normalise to [0-1]
  f[8]  = c2[1];
  f[9]  = c2[2];
}

}  // anonymous namespace

namespace ppf {

ModelSearch::ModelSearch(const Eigen::Matrix<float, 3, Eigen::Dynamic>& model_points,
                         const Eigen::Matrix<float, 3, Eigen::Dynamic>& model_normals,
                         boost::optional<Eigen::Matrix<float, 1, Eigen::Dynamic>> model_colors,
                         float distance_quantization_step, float angle_quantization_step,
                         std::vector<float> color_quantization_step, Spreading spreading,
                         size_t num_anchor_points, FeatureType ppf_type
                         )
: num_points_(model_points.cols()), num_anchor_points_(num_anchor_points), model_diameter_(-1.0f),
  distance_quantization_step_(distance_quantization_step), angle_quantization_step_(angle_quantization_step),
  color_quantization_step_(color_quantization_step), spreading_(spreading), num_features_(ppf_type == FeatureType::CPPF ? 10 : 4) {
  CHECK(distance_quantization_step > 0.0f);
  CHECK(angle_quantization_step > 0.0f);
  CHECK(num_anchor_points_ <= num_points_);

  if (num_anchor_points_ == 0)
    num_anchor_points_ = num_points_;

  LOG(INFO) << "Constructing ppf::ModelSearch object for a model with " << num_points_ << " points ("
            << num_anchor_points_ << " anchors)";

  if (model_colors.is_initialized()) {
      model_point_colors_.resize(num_points_);
      for (uint32_t j = 0; j < num_points_; ++j) {
        model_point_colors_[j] = model_colors.get()(0,j);
      }
  }

  // In this block we compute point pair feature and local coordinate for every pair of points in the model. The PPFs
  // are then quantized (and optionally spread). These data are inserted into a hash table, where quantized PPFs are the
  // keys and LCs are values.

  std::map<QuantizedPPF, LocalCoordinate::Vector> buckets;

  // Compute point pair features for every pair of points in the cloud
  for (uint32_t i = 0; i < num_anchor_points_; ++i) {
    for (uint32_t j = 0; j < num_points_; ++j) {
      if (i == j)
        continue;
      const auto& p1 = model_points.col(i);
      const auto& n1 = model_normals.col(i);
      const auto& p2 = model_points.col(j);
      const auto& n2 = model_normals.col(j);

      if(ppf_type == FeatureType::CPPF) {
          float c1 = (*model_colors).col(i).value();
          Eigen::Vector3i rgb1;
          floatColToRGB(c1, rgb1);

          float c2 = (*model_colors).col(j).value();
          Eigen::Vector3i rgb2;
          floatColToRGB(c2, rgb2);

          Eigen::Vector3f hsv1;
          Eigen::Vector3f hsv2;
          RGBtoHSV (rgb1,hsv1);
          RGBtoHSV (rgb2,hsv2);

          float cppf_f[10];
          computeCPPF(p1, n1, p2, n2, hsv1, hsv2, cppf_f);

          // Calculate alpha_m angle ([VLLM18], figure 4)
          Eigen::Affine3f transform_mg;
          LocalCoordinate::computeTransform(p1, n1, transform_mg);
          auto alpha_m = LocalCoordinate::computeAngle(transform_mg * p2);

          std::vector<float> cppf_f_vec(std::begin(cppf_f), std::end(cppf_f));
          for (const auto& key : quantizePPF(cppf_f_vec, spreading_ == Spreading::On)) {
              buckets[key].push_back({i, j, alpha_m});
          }
          if (model_diameter_ < cppf_f[3])
              model_diameter_ = cppf_f[3];
      } else
      {
          float ppf_f[4];
          computePPF(p1, n1, p2, n2, ppf_f);

          // Calculate alpha_m angle ([VLLM18], figure 4)
          Eigen::Affine3f transform_mg;
          LocalCoordinate::computeTransform(p1, n1, transform_mg);
          auto alpha_m = LocalCoordinate::computeAngle(transform_mg * p2);

          std::vector<float> ppf_f_vec(std::begin(ppf_f), std::end(ppf_f));

          for (const auto& key : quantizePPF(ppf_f_vec, spreading_ == Spreading::On)) {
              buckets[key].push_back({i, j, alpha_m});
          }
          if (model_diameter_ < ppf_f[3])
              model_diameter_ = ppf_f[3];
        }
    }
  }

  // In this block we collect all the different quantized PPFs that appeared in the model and create a new minimal
  // perfect hash function. This function will have no collisions on our set of quantized PPFs and will map them to the
  // interval [0 .. |keys| - 1].

  std::vector<QuantizedPPF> keys;
  keys.reserve(buckets.size());
  std::transform(buckets.begin(), buckets.end(), std::back_inserter(keys), [](const auto& p) { return p.first; });

  // CMPH will create a broken hash if there are less than two keys. And there must be something wrong with the model
  // anyway if this is the case.
  CHECK(keys.size() > 1);

  if(ppf_type == FeatureType::CPPF)
  {
      std::vector<std::array<u_char,10>> a;
      for(auto v : keys) {
          std::array<u_char,10> arr;
          for (size_t i = 0; i < 10; i++) {
              if(v[i] > 255) {
                  throw "Too many buckets! Can't cast to unsigned char";
              }
              arr[i] = (u_char)v[i];
          }
          //std::copy_n(v.begin(),10,arr.begin());
          a.push_back(arr);
      }

      //cmph_io_adapter_t* source = cmph_io_vector_adapter(a.data(),featureSize);
      cmph_io_adapter_t* source = cmph_io_struct_vector_adapter(a.data(), (cmph_uint32)sizeof(a[0]), 0, sizeof(a[0]), a.size());
      cmph_config_t* config = cmph_config_new(source);
      cmph_config_set_algo(config, CMPH_CHD);
      cmph_config_set_verbosity(config,10);
      cmph_config_set_b(config,6);
      hash_function_ = cmph_new(config);
      cmph_config_destroy(config);

      // The previously created hash function is perfect and minimal, therefore it's output can be used as a displacement
      // in a table. Here we rearrange LCs according to the indices output by the function. In addition to that, we create
      // a table to store quantized PPFs that yield corresponding indices. This is needed because CMPH hash function always
      // outputs some index, even for qPPFs that were not used to create it. Therefore after an index is found we need to
      // verify that the qPPF that is associated to it is indeed the one that we hashed.

      // The reason for +1: the very last entry of the table will remain empty. We will return a reference to it from the
      // find() method whenever there are no matches for the queried pair of points.
      lcs_table_.resize(buckets.size() + 1);
      qppf_table_.resize(buckets.size());
      for (auto& bucket : buckets) {
          std::array<u_char,10> arr;
          for (size_t i = 0; i < 10; i++) {
              if(bucket.first[i] > 255) {
                  throw "Too many buckets! Can't cast to unsigned char";
              }
              arr[i] = (u_char)bucket.first[i];
          }
          unsigned int id = cmph_search(hash_function_, reinterpret_cast<const char*>(&arr), sizeof(arr)); //search function needs key length in bytes
          std::swap(lcs_table_[id], bucket.second);  // these are vectors, swap
          qppf_table_[id] = bucket.first;            // these are 4-tuples, assign
      }
  }
  else
  {
      std::vector<std::array<u_char,4>> a;
      for(auto v : keys) {
          std::array<u_char,4> arr;
          for (size_t i = 0; i < 4; i++) {
              if(v[i] > 255) {
                  throw "Too many buckets! Can't cast to unsigned char";
              }
              arr[i] = (u_char)v[i];
          }
          //std::copy_n(v.begin(),4,arr.begin());
          a.push_back(arr);
      }


      auto size = sizeof(a[0]);

      //cmph_io_adapter_t* source = cmph_io_vector_adapter(a.data(),featureSize);
      cmph_io_adapter_t* source = cmph_io_struct_vector_adapter(a.data(), (cmph_uint32)sizeof(a[0]), 0, sizeof(a[0]), a.size());
      cmph_config_t* config = cmph_config_new(source);
      cmph_config_set_algo(config, CMPH_CHD);
      hash_function_ = cmph_new(config);
      cmph_config_destroy(config);

      // The previously created hash function is perfect and minimal, therefore it's output can be used as a displacement
      // in a table. Here we rearrange LCs according to the indices output by the function. In addition to that, we create
      // a table to store quantized PPFs that yield corresponding indices. This is needed because CMPH hash function always
      // outputs some index, even for qPPFs that were not used to create it. Therefore after an index is found we need to
      // verify that the qPPF that is associated to it is indeed the one that we hashed.

      // The reason for +1: the very last entry of the table will remain empty. We will return a reference to it from the
      // find() method whenever there are no matches for the queried pair of points.
      lcs_table_.resize(buckets.size() + 1);
      qppf_table_.resize(buckets.size());
      for (auto& bucket : buckets) {
          std::array<u_char,4> arr;
          for (size_t i = 0; i < 4; i++) {
              if(bucket.first[i] > 255) {
                  throw "Too many buckets! Can't cast to unsigned char";
              }
              arr[i] = (u_char)bucket.first[i];
          }

          unsigned int id = cmph_search(hash_function_, reinterpret_cast<const char*>(&arr),sizeof(arr)); //search function needs key length in bytes
          std::swap(lcs_table_[id], bucket.second);  // these are vectors, swap
          qppf_table_[id] = bucket.first;            // these are 4-tuples, assign
      }
  }


  // Due to spreading, hash table entries (i.e. vectors of local coordinates that correspond to a particular quantized
  // PPF) may contain very similar items. We want to eliminate such items because a) the less items there are, the
  // faster the voting loop will be in PPF recognition pipeline; b) votes from similar items are redundant anyway.

  for (auto& entry : lcs_table_) {
    // Compaction via local clustering
    struct Policy {
      const float step;
      using Object = LocalCoordinate;
      using Cluster = LocalCoordinate;
      bool similarToCluster(const Object& object, const Cluster& cluster) const {
        return object.model_point_index1 == cluster.model_point_index1 &&
               std::abs(std::remainder(cluster.rotation_angle - object.rotation_angle, 2.0 * M_PI)) < step;
      }
    };
    v4r::GreedyLocalClustering<Policy> glc(angle_quantization_step_);
    glc.add(entry.begin(), entry.end());
    auto clusters = glc.getClusters();

    // Also sort by model index
    std::sort(clusters.begin(), clusters.end(),
              [](const auto& a, const auto& b) { return a.model_point_index1 < b.model_point_index1; });
    // Move into the hash table
    std::swap(entry, clusters);
  }

  // The user should be able to convert LCs to SE(3) transforms. The transforms depend on the model point coordinates
  // and normals. To minimize memory footprint we compute and store translation and quaternion rotation instead of
  // full 4x4 transform matrix.

  lc_translations_.resize(num_points_);
  lc_rotations_.resize(num_points_);
  for (size_t i = 0; i < num_points_; ++i) {
    const auto& p = model_points.col(i);
    const auto& n = model_normals.col(i);
    LocalCoordinate::computeTransform(p, n, lc_translations_[i], lc_rotations_[i]);
  }

}

ModelSearch::ModelSearch(const std::string& filename) {
  LOG(INFO) << "Loading ppf::ModelSearch object from file " << filename;
  auto file = fopen(filename.c_str(), "rb");
  read(file, num_points_);
  read(file, num_anchor_points_);
  read(file, model_diameter_);
  read(file, distance_quantization_step_);
  read(file, angle_quantization_step_);
  read(file, color_quantization_step_); //this is new --> models need to be retrained!
  read(file, spreading_);

  model_point_colors_.resize(read<size_t>(file));
  for (auto& p : model_point_colors_)
      read(file, p);

  lcs_table_.resize(read<size_t>(file));
  for (auto& p : lcs_table_)
    read(file, p);

  qppf_table_.resize(read<size_t>(file));
  for(auto&p :qppf_table_)
      read(file,p);

  read(file, lc_translations_);
  read(file, lc_rotations_);
  hash_function_ = cmph_load(file);
  fclose(file);
}

ModelSearch::~ModelSearch() {
  if (hash_function_)
    cmph_destroy(hash_function_);
}

const LocalCoordinate::Vector& ModelSearch::find(const Eigen::Vector3f& p1, const Eigen::Vector3f& n1,
                                                 const Eigen::Vector3f& p2, const Eigen::Vector3f& n2) const {

  float ppf[4];
  computePPF(p1, n1, p2, n2, ppf);
  std::vector<float> ppf_vec(std::begin(ppf), std::end(ppf));
  auto qppf = quantizePPF(ppf_vec);

  std::array<u_char,4> arr;
  for (size_t i = 0; i < 4; i++) {
      if(qppf[i] > 255) {
          throw "Too many buckets! Can't cast to char";
      }
    arr[i] = (u_char)qppf[i];
  }
  //std::copy(qppf.begin(),qppf.end(),arr.begin());

  unsigned int id = cmph_search(hash_function_, reinterpret_cast<const char*>(&arr), sizeof(arr)); // 4 feature components
  if (id >= qppf_table_.size()) { //the hash function generated a value that was not seen during training
        return lcs_table_.back();
  }

  // Check if the query qPPF is the same as stored in the table under the found index. In not, it means that the queried
  // qPPF has no similarities in the model and we return an empty list of LCs (which is conveniently stored in the very
  // last entry of the LCs table).
  for (size_t i = 0; i < 4; ++i){
      auto v = qppf_table_[id];
      if (v.empty() || qppf[i] != v[i])
          return lcs_table_.back();
  }
  return lcs_table_[id];
}

const LocalCoordinate::Vector& ModelSearch::find(const Eigen::Vector3f& p1, const Eigen::Vector3f& n1,
                                                 const Eigen::Vector3f& p2, const Eigen::Vector3f& n2,
                                                 const Eigen::Vector3i& c1, const Eigen::Vector3i& c2) const {
    Eigen::Vector3f hsv1;
    Eigen::Vector3f hsv2;
    RGBtoHSV (c1,hsv1);
    RGBtoHSV (c2,hsv2);

    float cppf_f[10];
    computeCPPF(p1, n1, p2, n2, hsv1, hsv2, cppf_f);
    std::vector<float> ppf_vec(std::begin(cppf_f), std::end(cppf_f));
    auto qppf = quantizePPF(ppf_vec);
    auto da = qppf.data();

    std::array<u_char,10> arr;
    for (size_t i = 0; i < 10; i++) {
        if(qppf[i] > 255) {
            throw "Too many buckets! Can't cast to char";
        }
      arr[i] = (u_char)qppf[i];
    }
    //std::copy(qppf.begin(),qppf.end(),arr.begin());
    //std::cout << &arr << std::endl;

    unsigned int id = cmph_search(hash_function_, reinterpret_cast<const char*>(&arr), sizeof(arr)); // 10 feature components

    if (id >= qppf_table_.size()) { //the hash function generated a value that was not seen during training
          return lcs_table_.back();
    }
    // Check if the query qPPF is the same as stored in the table under the found index. If not, it means that the queried
    // qPPF has no similarities in the model and we return an empty list of LCs (which is conveniently stored in the very
    // last entry of the LCs table).
    for (size_t i = 0; i < 10; ++i){
        auto v = qppf_table_[id];
        if (v.empty() || qppf[i] != v[i])
            return lcs_table_.back();
    }
    return lcs_table_[id];
}

void ModelSearch::computeTransform(const LocalCoordinate& lc, Eigen::Affine3f& transform) const {
  transform = Eigen::AngleAxisf(lc.rotation_angle, Eigen::Vector3f::UnitX()) * lc_translations_[lc.model_point_index1] *
              lc_rotations_[lc.model_point_index1];
}

void ModelSearch::getModelPoints(Eigen::Map<Eigen::MatrixXf, Eigen::Aligned, Eigen::OuterStride<>> model_points) const {
  for (size_t i = 0; i < num_points_; ++i)
    model_points.col(i) = lc_rotations_[i].inverse() * -lc_translations_[i].translation();
}

void ModelSearch::getModelNormals(
    Eigen::Map<Eigen::MatrixXf, Eigen::Aligned, Eigen::OuterStride<>> model_normals) const {
  for (size_t i = 0; i < num_points_; ++i)
    model_normals.col(i) = lc_rotations_[i].inverse() * Eigen::Vector3f::UnitX();
}

void ModelSearch::save(const std::string& filename) const {
  LOG(INFO) << "Saving ppf::ModelSearch object to file " << filename;
  auto file = fopen(filename.c_str(), "wb");
  write(file, num_points_);
  write(file, num_anchor_points_);
  write(file, model_diameter_);
  write(file, distance_quantization_step_);
  write(file, angle_quantization_step_);
  write(file, color_quantization_step_);
  write(file, spreading_);

  write(file, model_point_colors_.size());
  for (const auto& p : model_point_colors_)
      write(file, p);

  auto size = lcs_table_.size();
  write(file, size);
  for (const auto& p : lcs_table_)
    write(file, p);

  write(file, qppf_table_.size());
  for(const auto& p : qppf_table_)
    write(file,p);

  write(file, lc_translations_);
  write(file, lc_rotations_);
  cmph_dump(hash_function_, file);
  fclose(file);
}

ModelSearch ModelSearch::load(const std::string& filename) {
  return ModelSearch(filename);
}


ModelSearch::QuantizedPPF ModelSearch::quantizePPF(std::vector<float> ppf) const {
  QuantizedPPF qppf;
  qppf.resize(ppf.size());
  for (size_t i = 0; i < 3; ++i)
    qppf[i] = std::trunc(ppf[i] / angle_quantization_step_);
  qppf[3] = std::trunc(ppf[3] / distance_quantization_step_);

  if (ppf.size() == 10) {
      for (size_t i = 0; i < 3; ++i) { //first color vector
          qppf[4+i] = std::trunc(ppf[4+i] / color_quantization_step_[i]);
      }
      for (size_t i = 0; i < 3; ++i) { //second color vector
          qppf[7+i] = std::trunc(ppf[7+i] / color_quantization_step_[i]);
      }
  }
  return qppf;
}

std::vector<ModelSearch::QuantizedPPF> ModelSearch::quantizePPF(std::vector<float> ppf, bool with_spreading) const {
  if (!with_spreading)
    return {quantizePPF(ppf)};

  std::vector<ModelSearch::QuantizedPPF> qppfs(1);
  qppfs[0] = std::vector<int32_t>(ppf.size());
  const std::array<float, 10> steps = {angle_quantization_step_, angle_quantization_step_, angle_quantization_step_,
                                      distance_quantization_step_, color_quantization_step_[0], color_quantization_step_[1], color_quantization_step_[2],
                                      color_quantization_step_[0], color_quantization_step_[1], color_quantization_step_[2]};
  std::vector<int> shifts(ppf.size());
  for (size_t i = 0; i < ppf.size(); ++i) {
    auto d = ppf[i] / steps[i];
    auto t = std::trunc(d);
    auto f = d - t;

    if (std::abs(f) < 1.0 / 3)
      shifts[i] = -1;
    else if (std::abs(f) < 2.0 / 3)
      shifts[i] = 0;
    else
      shifts[i] = 1;
    if (f < 0)
      shifts[i] *= -1;

    qppfs[0][i] = t;
  }

  for (size_t i = 0; i < ppf.size(); ++i) {
    if (shifts[i] != 0) {
      auto size = qppfs.size();
      for (size_t j = 0; j < size; ++j) {
        qppfs.push_back(qppfs[j]);
        qppfs.back()[i] += shifts[i];
      }
    }
  }

  // Wrap around angles such that they are in [0..π] range
  const auto max = std::trunc(M_PI / angle_quantization_step_);
  for (auto& qppf : qppfs)
    for (size_t i = 0; i < 3; ++i)
      if (qppf[i] < 0)
        qppf[i] = max;
      else if (qppf[i] > max)
        qppf[i] = 0;

  if (ppf.size() == 10) { //also wrap Hue angles
    const auto max = std::trunc(1 / color_quantization_step_[0]);
    for (auto& qppf : qppfs)
      for (size_t i = 4; i < 10 ; i+=3) //feature value 4 and 7 (start counting with 0) correspond to HUE, which is normalized between 0 and 1, but an angle
        if (qppf[i] < 0)
          qppf[i] = max;
        else if (qppf[i] > max)
          qppf[i] = 0;
  }

  // Remove qPPS that ended up having negative distance as this is not allowed.
  qppfs.erase(std::remove_if(qppfs.begin(), qppfs.end(), [](const auto& qppf) { return qppf[3] < 0; }), qppfs.end());
  if (ppf.size() == 10) { //remove S and V bin values that are negative or bigger than max_possible_bin
      int max_bin_saturation = std::trunc(1/color_quantization_step_[1]); //max saturation value is 1, find the max. possible bin
      int max_bin_value = std::trunc(1/color_quantization_step_[2]);
      qppfs.erase(std::remove_if(qppfs.begin(), qppfs.end(), [&max_bin_saturation](const auto& qppf) { return (qppf[5] < 0 || qppf[5] > max_bin_saturation); }), qppfs.end());
      qppfs.erase(std::remove_if(qppfs.begin(), qppfs.end(), [&max_bin_value](const auto& qppf) { return (qppf[6] < 0 || qppf[6] > max_bin_value); }), qppfs.end());
      qppfs.erase(std::remove_if(qppfs.begin(), qppfs.end(), [&max_bin_saturation](const auto& qppf) { return (qppf[8] < 0 || qppf[8] > max_bin_saturation); }), qppfs.end());
      qppfs.erase(std::remove_if(qppfs.begin(), qppfs.end(), [&max_bin_value](const auto& qppf) { return (qppf[9] < 0 || qppf[9] > max_bin_value); }), qppfs.end());
  }

  return qppfs;
}

}  // namespace ppf

