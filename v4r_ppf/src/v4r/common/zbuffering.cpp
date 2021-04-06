#include <v4r/common/point_types.h>
#include <pcl/impl/instantiate.hpp>
#include <v4r/common/impl/zbuffering.hpp>

namespace v4r {

#define PCL_INSTANTIATE_ZBuffering(T) template class ZBuffering<T>;
PCL_INSTANTIATE(ZBuffering, V4R_PCL_XYZ_POINT_TYPES)

}  // namespace v4r
