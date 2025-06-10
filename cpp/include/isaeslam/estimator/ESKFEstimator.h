#ifndef ESKFESTIMATOR_H
#define ESKFESTIMATOR_H
#include "isaeslam/estimator/APoseEstimator.h"

namespace isae {

/*!
 * @brief ESKFEstimator class for estimating the transformation between two frames using an EKF
 *
 * The estimation is done by perfoming an ESKF update on each landmark (the matches must be associated with landmarks).
 * This allows closed form computation of the covariance of the estimated transformation.
 */
class ESKFEstimator : public APoseEstimator {
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
     * @brief Refine the triangulation of the landmarks with an ESKF BEWARE: currently being tested
     */
    bool refineTriangulation(std::shared_ptr<Frame> &frame);
};

} // namespace isae
#endif // ESKFESTIMATOR_H