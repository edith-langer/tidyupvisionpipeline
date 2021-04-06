# Point-Pairwise-Features (ppf) pipeline extracted from the v4r library

## C++ demo file usage

```bash
cd build
./test_ppf_recognizer -m /path/to/ppf_models/
```

## dependencies

### yes

* cmake >= 3.10
* boost
* glog
* cmph
* eigen
* yaml-cpp
* pcl version >= ?
* v4r
  * v4r/apps/CloudSegmenter

  * v4r/common/Clustering
  * v4r/common/DataContainer
  * v4r/common/downsampler
  * v4r/common/greedy_local_clustering
  * v4r/common/miscellaneous
  * v4r/common/point_types
  * v4r/common/time
  * v4r/common/intrinsics
  * v4r/common/color_comparison
  * v4r/common/depth_outlines_params
  * v4r/common/graph_geometric_consistency
  * v4r/common/noise_models
  * v4r/common/depth_outlines
  * v4r/common/topsort_pruning

  * v4r/geometry/average
  * v4r/geometry/geometry
  * v4r/geometry/normals

  * v4r/io/eigen
  * v4r/io/filesystem

  * v4r/recognition/model
  * v4r/recognition/object_hypothesis
  * v4r/recognition/source
  * v4r/recognition/hypothesis_verification
  * v4r/recognition/hypothesis_verification_param
  * v4r/recognition/local_feature_matching
  * v4r/recognition/multiview_recognizer

  * v4r/segmentation/all_headers
  * v4r/segmentation/plane_extractor
  * v4r/segmentation/Plane_extractor_sac
  * v4r/segmentation/Plane_extractor_tile
  * v4r/segmentation/Plane_extractor_organized_multiplane
  * v4r/segmentation/segmenter
  * v4r/segmentation/segmenter_euclidean
  * v4r/segmentation/segmenter_conditional_euclidean
  * v4r/segmentation/segmenter_2d_connected_components
  * v4r/segmentation/segmenter_organized_connected_components
  * v4r/segmentation/smooth_Euclidean_segmenter
  * v4r/segmentation/types
  * v4r/segmentation/segmentation_utils
  * v4r/segmentation/plane_utils

  * v4r/registration/noise_model_based_cloud_integration

* pcl_1_8 - contains files that are not in (or do not have the same ABI) as the current pcl release

### maybe

* omp ??

### changes

* remove profiling
* remove symbol export macros
* remove recognition pipelines other than ppf
* remove any visualization support
* remove change detection
