#ifndef DOUBLESPHERE_H
#define DOUBLESPHERE_H

#include "isaeslam/data/sensors/ASensor.h"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/types.hpp>

namespace isae {

/*!
 * @brief An image sensor class that uses a double sphere model.
 *
 * The implmementation of the double sphere model is based on the paper:
 * "The Double Sphere Camera Model" by Usenko et al.
 * Available at https://arxiv.org/abs/1807.08957
 */
class DoubleSphere : public ImageSensor {

  public:
    DoubleSphere(const cv::Mat &image, Eigen::Matrix3d K, double alpha, double xi) : ImageSensor() {
        _calibration = K;
        _raw_data    = image.clone();
        _alpha       = alpha;
        _xi          = xi;
        _has_depth   = false;
        _mask        = cv::Mat(_raw_data.rows, _raw_data.cols, CV_8UC1, cv::Scalar(255));
    }

    bool project(const Eigen::Affine3d &T_w_lmk,
                 const std::shared_ptr<AModel3d> ldmk_model,
                 std::vector<Eigen::Vector2d> &p2ds) override;
    bool project(const Eigen::Affine3d &T_w_lmk,
                 const std::shared_ptr<AModel3d> ldmk_model,
                 const Eigen::Affine3d &T_f_w,
                 std::vector<Eigen::Vector2d> &p2ds) override;
    bool project(const Eigen::Affine3d &T_w_lmk,
                 const Eigen::Affine3d &T_f_w,
                 const Eigen::Matrix2d sqrt_info,
                 Eigen::Vector2d &p2d,
                 double *J_proj_frame,
                 double *J_proj_lmk) override;

    Eigen::Vector3d getRayCamera(Eigen::Vector2d f);
    Eigen::Vector3d getRay(Eigen::Vector2d f);
    double getFocal() override { return (_calibration(0, 0) + _calibration(1, 1)) / 2; }

  private:
    std::string _model;
    double _alpha; //!> Alpha parameter for the double sphere model
    double _xi;    //!> Xi parameter for the double sphere model
};

} // namespace isae

#endif