#pragma once

#include <pcl/point_cloud.h>
#include <v4r/common/depth_outlines.h>
#include <boost/dynamic_bitset.hpp>
#include <memory>

namespace v4r {

/**
 * Outline verification allows to check model outline against scene edges.
 * Internally it uses distance transform to compute Chamfer score which
 * is basically a root mean square average distance metric.
 * A loously reference paper behind that idea can be:
 * "Hierarchial Chamfer matching: A Parametric Edge Matching Algorithm" [Borgefors88]
 * https://people.eecs.berkeley.edu/~malik/cs294/borgefors88.pdf
 */
class OutlineVerification {
 public:
  explicit OutlineVerification(bool is_enabled = false) : is_enabled_(is_enabled) {}
  ~OutlineVerification() = default;

  /**
   * Computes distance transform for a scene for efficient chamfer score computation.
   * It is set only if outline verification is enabled.
   * @tparam T internal point cloud type
   * @param scene organized point cloud for a scene.
   */
  template <class T>
  void setScene(const pcl::PointCloud<T> &scene, DepthOutlinesParameter params = DepthOutlinesParameter()) {
    if (is_enabled_)
      depth_outlines_.reset(new DepthOutlines(scene, params));
  }

  /**
   * Clears scene
   */
  void resetScene() {
    depth_outlines_.reset();
  }

  /**
   * Turns off or on outline verification.
   * If it's off then model is never recognized as outlier and no computation happens.
   * @param is_enabled true if verification should be enabled
   */
  void setEnabled(bool is_enabled) {
    is_enabled_ = is_enabled;
  }

  /**
   *
   * @return true if outline verification is enabled
   */
  bool isEnabled() const {
    return is_enabled_;
  }

  /**
   * Just a getter for Chamfer score threshold value
   * @return current threshold
   */
  float getThreshold() const {
    return threshold_;
  }

  /**
   * Sets thershold value for Chamfer score. Values greater are discarded.
   * @param value
   */
  void setThreshold(float value) {
    threshold_ = value;
  }

  /**
   * Checks if model is outlier by comparing R.M.S Chamfer score against threshold.
   * @param model_image_mask image binary mask for an object
   * @param score optional pointer, if set score is write there
   * @return true if model is outlier
   */
  bool isOutlier(const boost::dynamic_bitset<> &model_image_mask, float *score = nullptr) const;

 private:
  float threshold_ = 4.75f;
  std::unique_ptr<DepthOutlines> depth_outlines_;
  bool is_enabled_ = false;
};

}  // namespace v4r
