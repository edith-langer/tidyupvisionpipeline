#include <glog/logging.h>
#include <v4r/common/time.h>

namespace v4r {

ScopeTime::~ScopeTime() {
  double val = this->getTime();
  VLOG(2) << title_ << " took " << val << " ms.";
}
}  // namespace v4r