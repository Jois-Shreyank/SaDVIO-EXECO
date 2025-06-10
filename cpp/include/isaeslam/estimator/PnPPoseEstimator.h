#ifndef PNPPOSEESTIMATOR_H
#define PNPPOSEESTIMATOR_H
#include "isaeslam/estimator/APoseEstimator.h"

namespace isae {

/*!
* @brief PnPPoseEstimator class for estimating the transformation between two frames using PnP
*
* The estimation is done by using the PnP algorithm on the matches between the two frames.
* The matches must be associated with 3D landmarks.
*/
class PnPPoseEstimator : public APoseEstimator {
  public:
    bool estimateTransformBetween(const std::shared_ptr<Frame> &frame1,
                                  const std::shared_ptr<Frame> &frame2,
                                  vec_match &matches,
                                  Eigen::Affine3d &dT,
                                  Eigen::MatrixXd &covdT) override;
    bool estimateTransformBetween(const std::shared_ptr<Frame> &frame1,
                                  const std::shared_ptr<Frame> &frame2,
                                  typed_vec_match &typed_matches,
                                  Eigen::Affine3d &dT,
                                  Eigen::MatrixXd &covdT) override;
};

} // namespace isae
#endif // PNPPOSEESTIMATOR_H
