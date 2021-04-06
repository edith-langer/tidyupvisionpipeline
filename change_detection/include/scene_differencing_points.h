#ifndef SCENEDIFFERENCINGPOINTS_H
#define SCENEDIFFERENCINGPOINTS_H

#include <memory>
#include <vector>

#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/search/search.h>
#include <pcl/segmentation/segment_differences.h>
#include <pcl/common/io.h>


class SceneDifferencingPoints
{
public:
    typedef pcl::PointXYZRGBNormal PointType;
    typedef pcl::PointCloud<PointType> Cloud;

    SceneDifferencingPoints();
    SceneDifferencingPoints(double thr);

    void setDistThreshold(double thr);

    Cloud::Ptr computeDifference(Cloud::Ptr scene1, Cloud::Ptr scene2, std::vector<int> &src_indices, std::vector<int> &corr_ind);

private:
    double thr;
};

#endif // SCENEDIFFERENCINGPOINTS_H
