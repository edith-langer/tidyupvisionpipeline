#include "color_histogram.h"


double ColorHistogram::colorCorr(pcl::PointCloud<PointNormal>::Ptr cloud1,
                                 pcl::PointCloud<PointNormal>::Ptr cloud2, int _nrBins) {
    int nrBins = _nrBins;
    cv::Mat hist1 = computeHSVHistogram(cloud1, nrBins);
    cv::Mat hist2 = computeHSVHistogram(cloud2, nrBins);

    double corr = cv::compareHist(hist1, hist2, cv::HISTCMP_CORREL);
    return corr;
}

//compute histogram based on HSV color space, taking only hue and saturation into account
cv::Mat ColorHistogram::computeHSVHistogram(pcl::PointCloud<PointNormal>::Ptr cloud, int _nrBins) {
    int maxVal_h = 360;
    int maxVal_s = 1;
    int nrBins = _nrBins;

//    int sizes[] = { nrBins, nrBins, nrBins };
//    cv::Mat hist = cv::Mat(3, sizes, CV_32FC1, cv::Scalar(0));

    int sizes[] = { nrBins, nrBins};
    cv::Mat hist = cv::Mat(2, sizes, CV_32FC1, cv::Scalar(0));

    for (unsigned i = 0; i < cloud->points.size(); i++) {
        if (!pcl::isFinite(cloud->points[i]))
            continue;

        const PointNormal &p = cloud->points.at(i);

        Eigen::Vector3i rgb;
        Eigen::Vector3f hsv;
        rgb << (int)p.r, (int)p.g, (int)p.b;
        RGBtoHSV(rgb, hsv);

        double bin1 = hsv[0] * (double)nrBins / maxVal_h;
        int bin_bucket1 = std::min((int)bin1, nrBins-1); //border case if binBucket=nrBins it is not a valid index
        double bin2 = hsv[1] * (double)nrBins / maxVal_s;
        int bin_bucket2 = std::min((int)bin2, nrBins-1);
        //double bin3 = hsv[2] * (double)nrBins / maxVal;
        //int bin_bucket3 = std::min((int)bin3, nrBins-1);

        //hist.at<float>(bin_bucket1, bin_bucket2, bin_bucket3) += 1;
        hist.at<float>(bin_bucket1, bin_bucket2) += 1;
    }

    float normalization = cloud->points.size() ;
    for (int i = 0; i < nrBins; i++) {
        for (int j = 0; j < nrBins; j++) {
            hist.at<float>(i,j) /= normalization;
        }
    }

    //std::cout << hist << std::endl;

    return hist;
}


void ColorHistogram::RGBtoHSV (const Eigen::Vector3i &in, Eigen::Vector3f &out)
{
  const unsigned char max = std::max (in[0], std::max (in[1], in[2]));
  const unsigned char min = std::min (in[0], std::min (in[1], in[2]));

  out[2] = static_cast <float> (max) / 255.f;

  if (max == 0) // division by zero
  {
    out[1] = 0.f;
    out[0] = 0.f; // h = -1.f;
    return;
  }

  const float diff = static_cast <float> (max - min);
  out[1] = diff / static_cast <float> (max);

  if (min == max) // diff == 0 -> division by zero
  {
    out[0] = 0;
    return;
  }

  if      (max == in[0]) out[0] = 60.f * (      static_cast <float> (in[1] - in[2]) / diff);
  else if (max == in[1]) out[0] = 60.f * (2.f + static_cast <float> (in[2] - in[0]) / diff);
  else                  out[0] = 60.f * (4.f + static_cast <float> (in[0] - in[1]) / diff); // max == b

  if (out[0] < 0.f) out[0] += 360.f;
}
