#ifndef REGIONGROWING_H
#define REGIONGROWING_H

#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/common/angles.h>
#include <pcl/octree/octree.h>
#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>

#include <v4r/common/color_comparison.h>


template <typename PointT, typename PointQ>
class RegionGrowing
{
public:
    RegionGrowing(typename pcl::PointCloud<PointT>::ConstPtr scene, typename pcl::PointCloud<PointQ>::ConstPtr object,
                  pcl::PointCloud<pcl::Normal>::ConstPtr scene_normals, bool is_object_downsampled,
                  float color_thr=std::numeric_limits<float>::max(),
                  double octree_res=0.05, float eps_angle_threshold_deg=10, float max_neighbour_distance=0.02, float curvature_threshold=0.08
                  ) :
        scene_(scene), object_(object), scene_normals_(scene_normals), is_object_downsampled_(is_object_downsampled),
        octree_res_(octree_res), eps_angle_threshold_deg_(eps_angle_threshold_deg),
        max_neighbour_distance_(max_neighbour_distance), curvature_threshold_(curvature_threshold), color_thr_(color_thr)
    {
    }

    std::vector<int> compute() {
//        typename pcl::PointCloud<PointT>::Ptr vis_cloud(new pcl::PointCloud<PointT>);
//        pcl::copyPointCloud(*scene_, *vis_cloud);
//        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
//        viewer.reset(new pcl::visualization::PCLVisualizer ("3D Viewer"));
//        viewer->setBackgroundColor (255, 255, 255);
//        pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(vis_cloud);
//        viewer->addPointCloud<PointT> (vis_cloud, rgb, "cloud");
//        viewer->resetCamera();
//        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "cloud");
//        viewer->initCameraParameters ();

        assert(scene_->points.size() == scene_normals_->points.size());

        /// create an octree for search
        octree_.reset(new pcl::octree::OctreePointCloudSearch<PointT>(octree_res_));
        octree_->setInputCloud(scene_);
        octree_->addPointsFromInputCloud();
        //std::cout << "Created octree" << std::endl;

        //  // Create a bool vector of processed point indices, and initialize it to false
        std::vector<bool> processed_scene(scene_->points.size(), false);
        std::vector<int> nn_indices;
        std::vector<float> nn_sqrt_distances;

        float eps_angle_threshold_rad = pcl::deg2rad(eps_angle_threshold_deg_);

        std::vector<int> orig_object_ind;
//        while (!viewer->wasStopped ()) {
        for (size_t i = 0; i < object_->points.size(); ++i) {
            if (!pcl::isFinite(object_->points[i]))
                continue;

            std::vector<int> seed_queue;
            int sq_idx = 0;

            PointT p_object;
            p_object.x = object_->points[i].x;
            p_object.y = object_->points[i].y;
            p_object.z = object_->points[i].z;

            octree_->nearestKSearch(p_object, 1, nn_indices, nn_sqrt_distances);
            if (nn_sqrt_distances[0] > max_neighbour_distance_*max_neighbour_distance_ || processed_scene[nn_indices[0]] || !pcl::isFinite(scene_->points[nn_indices[0]]))
                continue;

            const PointT query_pt = scene_->points[nn_indices[0]];
            const pcl::Normal query_n = scene_normals_->points[nn_indices[0]];

            seed_queue.push_back(nn_indices[0]);
            processed_scene[nn_indices[0]] = true;
            orig_object_ind.push_back(nn_indices[0]);

            if (is_object_downsampled_) {
                int closest_orig_ind = nn_indices[0];
                if (octree_->radiusSearch(p_object, std::sqrt(2) * 0.01, nn_indices, nn_sqrt_distances) > 0){ //we want to add all points within a radius of 1 cm to the result
                    for (size_t j = 0; j < nn_indices.size(); j++) {
                        if (processed_scene[nn_indices[j]] || !pcl::isFinite(scene_->points[nn_indices[j]]))
                            continue;
//                        if (nn_indices[j] == closest_orig_ind) { //only add the closest point to the downsampled point as seed for region growing
//                            seed_queue.push_back(nn_indices[j]);
//                            processed_scene[nn_indices[j]] = true;
//                        }
                        orig_object_ind.push_back(nn_indices[j]);
                    }
                }
            }

            while (sq_idx < seed_queue.size()) {
                int sidx = seed_queue[sq_idx];
                const PointT &query_pt = scene_->points[sidx];
                const pcl::Normal &query_n = scene_normals_->points[sidx];

                //                    if (query_n.curvature > curvature_threshold_) {
                //                        sq_idx++;
                //                        std::cout << "Query point high curvature" <<std::endl;
                //                        continue;
                //                    }

                float radius = max_neighbour_distance_;

                if (!octree_->radiusSearch(query_pt, std::sqrt(2) * radius, nn_indices, nn_sqrt_distances)) {
                    sq_idx++;
                    continue;
                }

                for (size_t j = 0; j < nn_indices.size(); j++) {
                    if (processed_scene[nn_indices[j]] || !pcl::isFinite(scene_->points[nn_indices[j]]))  // Has this point been processed before ?
                        continue;

//                    if (scene_normals_->points[nn_indices[j]].curvature > curvature_threshold_)
//                        //std::cout << "Neighbour point high curvature" <<std::endl;
//                        continue;

                    Eigen::Vector3f n1;
                    n1 = query_n.getNormalVector3fMap();

                    pcl::Normal nn = scene_normals_->points[nn_indices[j]];
                    const Eigen::Vector3f &n2 = nn.getNormalVector3fMap();

                    double dot_p = n1.dot(n2);

                    if (fabs(dot_p) > cos(eps_angle_threshold_rad)) {
                        if (color_thr_ != std::numeric_limits<float>::max()) {
                        //check if also color is similar
                            Eigen::Vector3i  m_rgb =  query_pt.getRGBVector3i();
                            Eigen::Vector3f color_m = rgb2lab(m_rgb);
                            Eigen::Vector3i  o_rgb =  scene_->points[nn_indices[j]].getRGBVector3i();
                            Eigen::Vector3f color_o = rgb2lab(o_rgb);
                            float color_distance_ = v4r::computeCIEDE2000(color_o, color_m);
                            if (color_distance_ > color_thr_) {
                                continue;
                            }
                        }
                        //add point
                        processed_scene[nn_indices[j]] = true;
                        seed_queue.push_back(nn_indices[j]);
                        orig_object_ind.push_back(nn_indices[j]);
//                        vis_cloud->points[nn_indices[j]].r =0;
//                        vis_cloud->points[nn_indices[j]].g =0;
//                        vis_cloud->points[nn_indices[j]].b =255;
                    }
                }
//                pcl::visualization::Camera cam;
//                viewer->getCameraParameters(cam);
//                viewer->removeAllPointClouds();
//                viewer->addPointCloud<PointT>(vis_cloud,"cloud");
//                viewer->setCameraParameters(cam);
//                viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "cloud");
//                for (size_t i =0; i < 100; i++ ) {
//                    usleep(100);
//                    viewer->spinOnce();

//                }
//                viewer->spinOnce();
                sq_idx++;
            }
        }
//        viewer->close();
//    }
        orig_object_ind.erase(std::unique(orig_object_ind.begin(), orig_object_ind.end()), orig_object_ind.end());
        return orig_object_ind;

    }
private:
    typename pcl::PointCloud<PointT>::ConstPtr scene_;
    typename pcl::PointCloud<PointQ>::ConstPtr object_;
    pcl::PointCloud<pcl::Normal>::ConstPtr scene_normals_; //normals of the scene cloud
    typename pcl::octree::OctreePointCloudSearch<PointT>::Ptr octree_;

    bool is_object_downsampled_;

    double octree_res_;
    float eps_angle_threshold_deg_;
    float max_neighbour_distance_;
    float curvature_threshold_;
    float color_thr_;

    Eigen::Vector3f rgb2lab(const Eigen::Vector3i &rgb) {
        cv::Mat rgb_cv (1,1, CV_8UC3);  //this has some information loss because Lab values are also just uchar and not float
        rgb_cv.at<cv::Vec3b>(0,0)[0] = rgb[0];
        rgb_cv.at<cv::Vec3b>(0,0)[1] = rgb[1];
        rgb_cv.at<cv::Vec3b>(0,0)[2] = rgb[2];
        cv::Mat lab_cv;
        cv::cvtColor(rgb_cv, lab_cv, CV_RGB2Lab);

        Eigen::Vector3f lab; //from opencv lab to orig lab definition
        lab << lab_cv.at<cv::Vec3b>(0,0)[0] / 2.55f, lab_cv.at<cv::Vec3b>(0,0)[1] - 128.f, lab_cv.at<cv::Vec3b>(0,0)[2] - 128.f;
        return lab;
    }
};

#endif // REGIONGROWING_H
