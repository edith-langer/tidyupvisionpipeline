#ifndef COLOR_HISTO_H
#define COLOR_HISTO_H

#include <opencv2/opencv.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

typedef pcl::PointXYZRGBNormal PointNormal;

class ColorHistogram
{
public:
    cv::Mat computeHSVHistogram(pcl::PointCloud<PointNormal>::Ptr cloud, int _nrBins);
    double colorCorr(pcl::PointCloud<PointNormal>::Ptr cloud1,
                                     pcl::PointCloud<PointNormal>::Ptr cloud2, int _nrBins=10);
private:
    void RGBtoHSV (const Eigen::Vector3i &in, Eigen::Vector3f &out);
};

#endif // COLOR_HISTO_H
