#ifndef AFEATURETRACKER_H
#define AFEATURETRACKER_H

#include <memory>
#include <utility>
#include <vector>

#include "isaeslam/data/sensors/Camera.h"
#include "isaeslam/featuredetectors/aFeatureDetector.h"
#include "isaeslam/typedefs.h"

namespace isae {

/*!
 * @brief Implements feature tracking between two images
 *
 * Contrary to feature matching, feature tracking doesn't require a set of features on the second image.
 * It uses optical flow to estimate the motion of features from the first image to the second image.
 */

class AFeatureTracker {
  public:
    AFeatureTracker() {}
    AFeatureTracker(std::shared_ptr<AFeatureDetector> detector) : _detector(detector) {}

    /*!
     * @brief Track features between two sensors
     *
     * @param sensor1 First sensor containing the features to track
     * @param sensor2 Second sensor where the features will be tracked
     * @param features_to_track Vector of features to track from sensor1
     * @param features_init Vector of features initialized in sensor1
     * @param tracks Output vector to store matched features
     * @param tracks_with_ldmk Output vector to store matched features with landmarks
     * @param search_width Width of the search area for tracking (default is 21)
     * @param search_height Height of the search area for tracking (default is 21)
     * @param nlvls_pyramids Number of pyramid levels for tracking (default is 3)
     * @param max_err Maximum error for tracking (default is 10)
     * @param backward Whether to track features backward to filter outliers (default is false)
     */
    virtual uint track(std::shared_ptr<isae::ImageSensor> &sensor1,
                       std::shared_ptr<isae::ImageSensor> &sensor2,
                       std::vector<std::shared_ptr<AFeature>> &features_to_track,
                       std::vector<std::shared_ptr<AFeature>> &features_init,
                       vec_match &tracks,
                       vec_match &tracks_with_ldmk,
                       int search_width   = 21,
                       int search_height  = 21,
                       int nlvls_pyramids = 3,
                       double max_err     = 10,
                       bool backward      = false) = 0;

  protected:
    std::shared_ptr<AFeatureDetector> _detector; //!< feature detector for feature init
    std::string _feature_label;                  //!< label for the features
};

} // namespace isae

#endif // AFEATURETRACKER_H
