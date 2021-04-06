#include <pcl/impl/instantiate.hpp>
#include <v4r/common/impl/pcl_opencv.hpp>

namespace v4r {
#define PCL_INSTANTIATE_PCLOpenCVConverter(T) template class  PCLOpenCVConverter<T>;
PCL_INSTANTIATE(PCLOpenCVConverter, V4R_PCL_RGB_POINT_TYPES)
}  // namespace v4r
