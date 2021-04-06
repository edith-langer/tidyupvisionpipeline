#include <v4r/recognition/outline_verification.h>

#include <opencv2/imgproc.hpp>

#include <opencv2/highgui.hpp>

#include <glog/logging.h>

namespace v4r {

namespace {

cv::Mat renderModelSilhouette(const boost::dynamic_bitset<> &model_image_mask, int width, int height) {
  cv::Mat model(height, width, CV_8UC1);

  for (size_t i = 0; i < model_image_mask.size(); ++i) {
    model.at<uint8_t>(i) = model_image_mask.test(i) ? 255 : 0;
  }

  // todo: just use other method to find edges
  cv::Canny(model, model, 100, 200, 3);
  cv::threshold(model, model, 10, 255, cv::THRESH_BINARY);

  return model;
}

}  // namespace

bool OutlineVerification::isOutlier(const boost::dynamic_bitset<> &model_image_mask, float *score) const {
  if (score)
    *score = 0.f;

  if (!depth_outlines_ || !is_enabled_)
    return false;

  cv::Mat model_silhouette =
      renderModelSilhouette(model_image_mask, depth_outlines_->getWidth(), depth_outlines_->getHeight());

  std::vector<cv::Point2i> locations;
  cv::findNonZero(model_silhouette, locations);

  const std::vector<float> distances = depth_outlines_->extractDistancesSquared(locations);
  float sum = std::accumulate(distances.begin(), distances.end(), 0.f);

  float chamfer_score =
      distances.empty() ? std::numeric_limits<float>::max() : (1.f / 3.f) * std::sqrt(sum / distances.size());
  VLOG(2) << "outline chamfer rms score: " << chamfer_score << "\n";

  if (score)
    *score = chamfer_score;

  return chamfer_score > threshold_;
}

}  // namespace v4r
