#ifndef CVKAZEFEATUREDETECTOR_H
#define CVKAZEFEATUREDETECTOR_H

#include "isaeslam/featuredetectors/aOpenCVFeatureDetector.h"

namespace isae {

/*!
 * @brief Class for detecting and computing KAZE features using OpenCV.
 */
class cvKAZEFeatureDetector : public AOpenCVFeatureDetector {
  public:
    cvKAZEFeatureDetector(int n, int n_per_cell, double max_matching_dist = 0.5)
        : AOpenCVFeatureDetector(n, n_per_cell) {
        _max_matching_dist = max_matching_dist;
        this->init();
    }

    void init() override;
    double computeDist(const cv::Mat &desc1, const cv::Mat &desc2) const override;
};

} // namespace isae

#endif // CVKAZEFEATUREDETECTOR_H