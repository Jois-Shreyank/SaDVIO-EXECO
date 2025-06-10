#ifndef EPIPOLARPOSEESTIMATOR_H
#define EPIPOLARPOSEESTIMATOR_H

#include "isaeslam/estimator/APoseEstimator.h"


namespace isae {

/*!
* @brief EpipolarPoseEstimator class for estimating the pose between two frames using Essential Matrix.
*
* Beware: the pose estimated with Essential matrix is up to a scale factor.
*/
class EpipolarPoseEstimator : public APoseEstimator {
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
    /*!
    * @brief A particular implementation of the pose estimation between two sensors.
    *
    * It was implemented for the non overlapping FoV case.
    */
    bool estimateTransformSensors(const std::shared_ptr<ImageSensor> &sensor1,
                                  const std::shared_ptr<ImageSensor> &sensor2,
                                  vec_match &matches,
                                  Eigen::Affine3d &dT,
                                  Eigen::MatrixXd &covdT);
};

} // namespace isae

#endif // EPIPOLARPOSEESTIMATOR_H