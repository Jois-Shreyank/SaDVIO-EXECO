#ifndef CAMERA_H
#define CAMERA_H

#include "isaeslam/data/sensors/ASensor.h"

namespace isae {

class Camera : public ImageSensor {

  public:
    Camera(const cv::Mat &image, Eigen::Matrix3d K) : ImageSensor() {
        _calibration = K;
        _raw_data    = image.clone();
        _mask        = cv::Mat(_raw_data.rows, _raw_data.cols, CV_8UC1, cv::Scalar(255));
        _has_depth   = false;
    }

    Camera(const cv::Mat &image, const cv::Mat mask, Eigen::Matrix3d K) : ImageSensor() {
        _calibration = K;
        _raw_data    = image.clone();
        _mask        = mask;
        _has_depth   = false;
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

    const cv::Mat &getDepthMat() override { return _raw_data; }
    std::vector<Eigen::Vector3d> getP3Dcam(const std::shared_ptr<AFeature> &feature) override {
        std::vector<Eigen::Vector3d> p3ds;
        return p3ds;
    }
};

} // namespace isae

#endif
