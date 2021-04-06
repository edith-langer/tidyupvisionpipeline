#pragma once

#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace v4r {
namespace io {

 Eigen::Matrix4f readMatrixFromFile(const boost::filesystem::path &file, size_t padding = 0);
 bool writeMatrixToFile(const boost::filesystem::path &, const Eigen::Matrix4f &);
}  // namespace io
}  // namespace v4r
