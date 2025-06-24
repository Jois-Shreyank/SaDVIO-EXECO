#ifndef FISHEYE_H
#define FISHEYE_H

#include "isaeslam/data/sensors/ASensor.h"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/types.hpp>

namespace isae {

/*!
 * @brief An image sensor class that uses the normalized spherical model.
 */
class Fisheye : public ImageSensor {

  public:
    Fisheye(const cv::Mat &image, Eigen::Matrix3d K, std::string model, float rmax)
        : ImageSensor(), _model(model), _rmax(rmax) {
        _calibration = K;
        _raw_data    = image.clone();
        _mask        = cv::Mat(_raw_data.rows, _raw_data.cols, CV_8UC1, cv::Scalar(255));
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
    double getFocal() override { return _rmax; }

  private:
    std::string _model;
    float _rmax;
};

/*!
 * @brief An image sensor class that uses the Omni model.
 *
 * The implementation of the Omni model is based on the paper:
 * "Single View Point Omnidirectional Camera Calibration from Planar Grids" by Mei et al.
 * Available at https://www.robots.ox.ac.uk/~cmei/articles/single_viewpoint_calib_mei_07.pdf
 */
class Omni : public ImageSensor {

  public:
    Omni(const cv::Mat &image, Eigen::Matrix3d K, double xi) : ImageSensor(), _xi(xi) {
        _calibration = K;
        _alpha       = _xi / (1 + _xi);
        _raw_data    = image.clone();
        _mask        = cv::Mat(_raw_data.rows, _raw_data.cols, CV_8UC1, cv::Scalar(255));
        _has_depth   = false;
        _distortion  = false;
    }

    Omni(const cv::Mat &image, Eigen::Matrix3d K, double xi, Eigen::Vector4d D) : ImageSensor(), _xi(xi) {
        _calibration = K;
        _alpha       = _xi / (1 + _xi);
        _raw_data    = image.clone();
        _mask        = cv::Mat(_raw_data.rows, _raw_data.cols, CV_8UC1, cv::Scalar(255));
        _has_depth   = false;
        _distortion  = true;
        _D           = D;
    }

    /*!
     * @brief Project a landmark in the image plane using the Omni model.
     *
     * This function is inspired by VINS-Mono implementation
     * Source :
     * https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/master/camera_model/src/camera_models/CataCamera.cc
     */
    bool project(const Eigen::Affine3d &T_w_lmk,
                 const std::shared_ptr<AModel3d> ldmk_model,
                 std::vector<Eigen::Vector2d> &p2ds) override;

    /*!
     * @brief Project a landmark in the image plane using the Omni model with frame pose.
     *
     * This function is inspired by VINS-Mono implementation
     * Source :
     * https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/master/camera_model/src/camera_models/CataCamera.cc
     */
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

    /*!
     * @brief Get the ray in camera coordinates from a 2D point in the image plane.
     *
     * This function is inspired by VINS-Mono implementation
     * Source :
     * https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/master/camera_model/src/camera_models/CataCamera.cc
     */
    Eigen::Vector3d getRayCamera(Eigen::Vector2d f);
    Eigen::Vector3d getRay(Eigen::Vector2d f);
    double getFocal() override { return (_calibration(0, 0) + _calibration(1, 1)) / 2; }

    /*!
     * @brief Distort a point using the omni distortion parameters.
     *
     * This function is inspired by VINS-Mono implementation
     * Source :
     * https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/master/camera_model/src/camera_models/CataCamera.cc
     */
    Eigen::Vector2d distort(const Eigen::Vector2d &p);

  private:
    double _xi, _alpha; //!< Xi and alpha parameters for the Omni model
    bool _distortion;   //!< Flag to indicate if distortion parameters are used
    Eigen::Vector4d _D; //!< Distortion parameters (k1, k2, p1, p2)
};

} // namespace isae

#endif
