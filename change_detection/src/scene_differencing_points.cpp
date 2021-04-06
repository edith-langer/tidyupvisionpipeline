#include "scene_differencing_points.h"

SceneDifferencingPoints::SceneDifferencingPoints()
{
    thr = 0.01;
}

SceneDifferencingPoints::SceneDifferencingPoints(double thr)
{
    this->thr = thr;
}

void SceneDifferencingPoints::setDistThreshold(double thr) {
    this->thr = thr;
}

/**
 * @brief computes the difference between two point clouds
 * @param scene1
 * @param scene2
 * @return a point cloud containing points that are in scene1, but not in scene2
 */
SceneDifferencingPoints::Cloud::Ptr SceneDifferencingPoints::computeDifference(Cloud::Ptr scene1, Cloud::Ptr scene2, std::vector<int> &src_indices, std::vector<int> &corr_ind) {
    Cloud::Ptr source = scene1;
    Cloud::Ptr target = scene2;
    Cloud::Ptr diff(new Cloud());

    // We're interested in a single nearest neighbor only
    std::vector<int> nn_indices (1);
    std::vector<float> nn_distances (1);

    pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>);
    tree->setInputCloud(target);

    // Iterate through the source data set
    for (int i = 0; i < static_cast<int> (source->points.size ()); ++i)
    {
        // Ignore invalid points in the inpout cloud
        if (!isFinite (source->points[i]))
            continue;
        // Search for the closest point in the target data set (number of neighbors to find = 1)
        //returns squared radius distances
        if (!tree->nearestKSearch (source->points[i], 1, nn_indices, nn_distances))
        {
            PCL_WARN ("No neighbor found for point %lu (%f %f %f)!\n", i, source->points[i].x, source->points[i].y, source->points[i].z);
            continue;
        }
        // Add points without a corresponding point in the target cloud to the output cloud
        if (std::sqrt(nn_distances[0]) > thr)
            src_indices.push_back (i);
        else
            corr_ind.push_back(nn_indices[0]);
    }

    // Copy all the data fields from the input cloud to the output one
    pcl::copyPointCloud(*source, src_indices, *diff);

    //TODO: what about the other way around?
    std::cout << "Completed computing difference of two scenes" << std::endl;

    return diff;
}
