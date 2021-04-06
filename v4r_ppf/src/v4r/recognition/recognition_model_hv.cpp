#include <v4r/recognition/object_hypothesis.h>

namespace v4r {

namespace {

constexpr bool kEnableDebugMaskDump = false;

void dumpMaskToFile(const std::string& file, int n, boost::dynamic_bitset<>& mask) {
  std::stringstream fn;
  fn << file << n << ".txt";
  std::ofstream f(fn.str().c_str());
  for (int i = 0, n = mask.size(); i < n; ++i) {
    f << mask[i] << " ";
  }
  f.close();
}

}  // namespace

template <typename ModelT>
void HVRecognitionModel<ModelT>::processSilhouette(bool do_smoothing, int smoothing_radius, bool do_erosion,
                                                   int erosion_radius, int img_width) {
  // EASY_BLOCK("Process silhouette");
  if (!do_smoothing && !do_erosion)
    return;

  for (int view = 0, n = image_mask_.size(); view < n; ++view) {
    auto& mask = image_mask_[view];
    int img_height = mask.size() / img_width;

    if (kEnableDebugMaskDump)
      dumpMaskToFile("/tmp/rendered_image_", view, mask);

    auto mask_has_in_radius = [&](bool value, int radius, int u, int v) -> bool {
      for (int vv = std::max(v - radius, 0); vv < std::min(v + radius, img_height); ++vv) {
        for (int uu = std::max(u - radius, 0); uu < std::min(u + radius, img_width); ++uu) {
          if (mask.test(vv * img_width + uu) == value)
            return true;
        }
      }
      return false;
    };

    if (do_smoothing) {
      boost::dynamic_bitset<> img_mask_smooth = mask;
      for (int v = 0; v < img_height; v++) {
        const int vw = v * img_width;
        for (int u = 0; u < img_width; u++) {
          if (mask_has_in_radius(true, smoothing_radius, u, v)) {
            img_mask_smooth.set(vw + u);
          }
        }
      }
      mask.swap(img_mask_smooth);

      if (kEnableDebugMaskDump)
        dumpMaskToFile("/tmp/rendered_image_smooth_", view, mask);
    }

    if (do_erosion) {
      boost::dynamic_bitset<> img_mask_eroded = mask;
      for (int v = 0; v < img_height; v++) {
        const int vw = v * img_width;
        for (int u = 0; u < img_width; u++) {
          if (mask_has_in_radius(false, erosion_radius, u, v)) {
            img_mask_eroded.set(vw + u, false);
          }
        }
      }
      mask.swap(img_mask_eroded);
      if (kEnableDebugMaskDump)
        dumpMaskToFile("/tmp/rendered_image_eroded_", view, mask);
    }
  }
}

template class HVRecognitionModel<pcl::PointXYZRGB>;
template class HVRecognitionModel<pcl::PointXYZ>;
}  // namespace v4r
