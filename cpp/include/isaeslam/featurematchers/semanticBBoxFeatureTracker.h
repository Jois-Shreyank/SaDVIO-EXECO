#ifndef SEMANTICBBOXFEATURETRACKER_H
#define SEMANTICBBOXFEATURETRACKER_H

#include "isaeslam/typedefs.h"
#include <type_traits>

#include "isaeslam/featurematchers/afeaturetracker.h"
#include "isaeslam/featurematchers/semanticBBoxFeatureMatcher.h"
namespace isae {

/*!
 * @brief Class for tracking 2D bounding boxes features
 */
class semanticBBoxFeatureTracker : public AFeatureTracker {
  public:
    semanticBBoxFeatureTracker() {}
    semanticBBoxFeatureTracker(std::shared_ptr<AFeatureDetector> detector)
        : AFeatureTracker(detector), _termCrit(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 20, 0.01) {
        _matcher = std::make_shared<semanticBBoxFeatureMatcher>(_detector);
    }

    uint track(std::shared_ptr<isae::ImageSensor> &sensor1,
               std::shared_ptr<isae::ImageSensor> &sensor2,
               std::vector<std::shared_ptr<AFeature>> &features_to_track,
               std::vector<std::shared_ptr<AFeature>> &features_init,
               vec_match &tracks,
               vec_match &tracks_with_ldmk,
               int search_width   = 21,
               int search_height  = 21,
               int nlvls_pyramids = 3,
               double max_err     = 10,
               bool backward      = false) override;

  private:
    std::shared_ptr<semanticBBoxFeatureMatcher> _matcher;

    cv::TermCriteria _termCrit;      //!< termination criteria for the optical flow algorithm
    double _klt_max_err       = 50.; //!< maximum error for KLT tracking
    double _max_backward_dist = 2.5; //!< maximum distance for backward tracking
};

} // namespace isae

#endif // SEMANTICBBOXFEATURETRACKER_H
