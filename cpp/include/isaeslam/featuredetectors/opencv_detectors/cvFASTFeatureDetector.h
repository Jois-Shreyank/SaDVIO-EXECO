#ifndef CVFASTFEATUREDETECTOR_H
#define CVFASTFEATUREDETECTOR_H

#include "isaeslam/featuredetectors/aOpenCVFeatureDetector.h"

namespace isae {

/*!
 * @brief Class for detecting and computing FAST features using OpenCV.
 */
class cvFASTFeatureDetector : public AOpenCVFeatureDetector {
  public:
    cvFASTFeatureDetector(int n, int n_per_cell, double max_matching_dist = 64)
        : AOpenCVFeatureDetector(n, n_per_cell) {
        _max_matching_dist = max_matching_dist;
        this->init();
    }

    void init() override;
    double computeDist(const cv::Mat &desc1, const cv::Mat &desc2) const override;
};

} // namespace isae

#endif // CVFASTFEATUREDETECTOR_H