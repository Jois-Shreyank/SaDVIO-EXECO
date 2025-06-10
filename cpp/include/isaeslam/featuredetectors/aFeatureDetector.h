#ifndef AFEATUREDETECTOR_H
#define AFEATUREDETECTOR_H

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "isaeslam/data/features/Line2D.h"
#include "isaeslam/data/features/Point2D.h"
#include "isaeslam/data/sensors/ASensor.h"
#include "isaeslam/typedefs.h"
#include "utilities/geometry.h"

namespace isae {

class Point2D;

/*!
 * @brief Abstract class for feature detectors.
 *
 * This class defines the interface for feature detection and descriptor computation.
 */
class AFeatureDetector {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    AFeatureDetector() {}
    AFeatureDetector(int n, int n_per_cell) : _n_total(n), _n_per_cell(n_per_cell) {}
    ~AFeatureDetector() {}

    /*!
     * @brief Virtual function to initialize the feature detector.
     *
     * This method should be called before using the detector.
     */
    virtual void init() = 0;

    /*!
     * @brief Virtual function to detect and compute features in an image.
     *
     * @param img The input image in which to detect features.
     * @param mask An optional mask to specify regions of interest in the image.
     * @param keypoints Output vector to store detected keypoints.
     * @param descriptors Output matrix to store computed descriptors.
     * @param n_points The number of points to detect (optional, default is 0 which means all).
     */
    virtual void detectAndCompute(const cv::Mat &img,
                                  const cv::Mat &mask,
                                  std::vector<cv::KeyPoint> &keypoints,
                                  cv::Mat &descriptors,
                                  int n_points = 0) = 0;

    /*!
     * @brief Virtual function to compute descriptors for a set of features.
     *
     * @param img The input image from which to compute descriptors.
     * @param features A vector of features for which to compute descriptors.
     */
    virtual void computeDescriptor(const cv::Mat &img, std::vector<std::shared_ptr<AFeature>> &features) = 0;

    /*!
     * @brief Virtual function to detect and compute features in a grid (bucketting).
     *
     * @param img The input image in which to detect features.
     * @param mask An optional mask to specify regions of interest in the image.
     * @param existing_features A vector of existing features to consider (optional).
     */
    virtual std::vector<std::shared_ptr<AFeature>> detectAndComputeGrid(
        const cv::Mat &img,
        const cv::Mat &mask,
        std::vector<std::shared_ptr<AFeature>> existing_features = std::vector<std::shared_ptr<AFeature>>()) = 0;

    /*!
     * @brief Virtual function to compute the distance between two feature descriptors.
     *
     * @param desc1 The first feature descriptor.
     * @param desc2 The second feature descriptor.
     */
    virtual double computeDist(const cv::Mat &desc1, const cv::Mat &desc2) const = 0;

    size_t getNbDesiredFeatures() { return _n_total; }
    double getMaxMatchingDist() const { return _max_matching_dist; }

    /*!
     * @brief Get features from a feature set in a bounding box defined by (x, y, w, h).
     * 
     * @param x The x-coordinate of the top-left corner of the bounding box.
     * @param y The y-coordinate of the top-left corner of the bounding box.
     * @param w The width of the bounding box.
     * @param h The height of the bounding box.
     * @param features The vector of features to search in.
     * @param features_in_box The vector to store the features found in the bounding box.
     */
    bool getFeaturesInBox(int x,
                          int y,
                          int w,
                          int h,
                          const std::vector<std::shared_ptr<AFeature>> &features,
                          std::vector<std::shared_ptr<AFeature>> &features_in_box) const;

    void deleteUndescribedFeatures(std::vector<std::shared_ptr<AFeature>> &features);

    /*!
     * @brief Convert OpenCV keypoints and descriptors to a vector of AFeature pointers.
     *
     * @param keypoints The input vector of OpenCV keypoints.
     * @param descriptors The input matrix of descriptors corresponding to the keypoints.
     * @param features Output vector to store the converted AFeature pointers.
     * @param featurelabel Optional label for the features (default is "pointxd").
     */
    static void KeypointToFeature(std::vector<cv::KeyPoint> keypoints,
                                  cv::Mat descriptors,
                                  std::vector<std::shared_ptr<AFeature>> &features,
                                  const std::string &featurelabel = "pointxd");

    /*!
     * @brief Convert a vector of AFeature pointers to OpenCV keypoints and descriptors.
     *
     * @param features The input vector of AFeature pointers.
     * @param keypoints Output vector to store the converted OpenCV keypoints.
     * @param descriptors Output matrix to store the converted descriptors.
     */
    static void FeatureToKeypoint(std::vector<std::shared_ptr<AFeature>> features,
                                  std::vector<cv::KeyPoint> &keypoints,
                                  cv::Mat &descriptors);

    /*!
     * @brief Convert a vector of AFeature pointers to a vector of cv::Point2f.
     *
     * @param features The input vector of AFeature pointers.
     * @param p2fs Output vector to store the converted cv::Point2f points.
     */
    static void FeatureToP2f(std::vector<std::shared_ptr<AFeature>> features, std::vector<cv::Point2f> &p2fs);

  protected:
    int _n_total;              //!> the maximum amount of features the detector should find for any given image
    int _n_per_cell;           //!> the number of features per cell
    double _max_matching_dist; //!> distance threshold for matching
};

} // namespace isae

#endif // AFEATUREDETECTOR_H
