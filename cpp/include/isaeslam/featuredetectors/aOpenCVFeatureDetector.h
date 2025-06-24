#ifndef AOPENCVFEATUREDETECTOR_H
#define AOPENCVFEATUREDETECTOR_H

#include "isaeslam/data/features/Point2D.h"
#include "isaeslam/data/sensors/ASensor.h"
#include "isaeslam/featuredetectors/aFeatureDetector.h"
#include "isaeslam/typedefs.h"

namespace isae {

class Point2D;

/*!
 * @brief AOpenCVFeatureDetector class for detecting and computing features using OpenCV.
 */
class AOpenCVFeatureDetector : public AFeatureDetector {
  public:
    AOpenCVFeatureDetector(int n, int n_per_cell) : AFeatureDetector(n, n_per_cell) {}

    void detectAndCompute(const cv::Mat &img,
                          const cv::Mat &mask,
                          std::vector<cv::KeyPoint> &keypoints,
                          cv::Mat &descriptors,
                          int n_points = 0);
    void computeDescriptor(const cv::Mat &img, std::vector<std::shared_ptr<AFeature>> &features);

    /*!
    * @brief Retain the n best keypoints based on their response
    */
    void retainBest(std::vector<cv::KeyPoint> &_keypoints, int n);
    std::vector<std::shared_ptr<AFeature>> detectAndComputeGrid(
        const cv::Mat &img,
        const cv::Mat &mask,
        std::vector<std::shared_ptr<AFeature>> existing_features = std::vector<std::shared_ptr<AFeature>>());

  protected:
    cv::Ptr<cv::FeatureDetector> _detector;       //!< Stores the opencv detector
    cv::Ptr<cv::DescriptorExtractor> _descriptor; //!< Stores the opencv descriptor extractor
};

} // namespace isae

#endif // AOPENCVFEATUREDETECTOR_H
