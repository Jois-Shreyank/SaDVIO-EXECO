#ifndef BUNDLEADJUSTMENTCERESNUMERIC_H
#define BUNDLEADJUSTMENTCERESNUMERIC_H

#include "isaeslam/optimizers/AOptimizer.h"

namespace isae {

class BundleAdjustmentCERESNumeric : public AOptimizer {
  public:
    BundleAdjustmentCERESNumeric()  = default;
    ~BundleAdjustmentCERESNumeric() = default;

  protected:
    uint addResidualsLocalMap(ceres::Problem &problem,
                              ceres::LossFunction *loss_function,
                              ceres::ParameterBlockOrdering *ordering,
                              std::vector<std::shared_ptr<Frame>> &frame_vector,
                              size_t fixed_frame_number,
                              std::shared_ptr<isae::LocalMap> &local_map) override;

    uint addSingleFrameResiduals(ceres::Problem &problem,
                                 ceres::LossFunction *loss_function,
                                 std::shared_ptr<Frame> &frame,
                                 typed_vec_landmarks &cloud_to_optimize) override;

    uint addLandmarkResiduals(ceres::Problem &problem,
                              ceres::LossFunction *loss_function,
                              typed_vec_landmarks &cloud_to_optimize) override;

  private:

    struct ReprojectionErrCeres_pointxd {

        ReprojectionErrCeres_pointxd(const Eigen::Vector2d &p2d,
                                     const std::shared_ptr<ImageSensor> &cam,
                                     const Eigen::Affine3d &T_w_lmk,
                                     const double sigma = 1.0)
            : _p2d(p2d), _cam(cam), _T_w_lmk(T_w_lmk) {
            _info_sqrt = (1 / sigma) * Eigen::Matrix2d::Identity();
        }

        // Constant parameters used to process the residual
        const Eigen::Vector2d _p2d;
        const std::shared_ptr<ImageSensor> _cam;
        const Eigen::Affine3d _T_w_lmk;
        Eigen::Matrix2d _info_sqrt;

        template <typename T> bool operator()(const T *const dX, const T *const dlmk, T *residual) const {

            // Get World to sensor transform
            Eigen::Affine3d T_f_w = _cam->getFrame()->getWorld2FrameTransform() * geometry::se3_doubleVec6dtoRT(dX);

            // Get Landmark P3D pose
            Eigen::Affine3d T_w_lmk = _T_w_lmk * geometry::se3_doubleVec3dtoRT(dlmk);

            Eigen::Vector2d projection;
            Eigen::Map<Eigen::Vector2d> res(residual);

            if (!_cam->project(T_w_lmk, T_f_w, _info_sqrt, projection, NULL, NULL)) {
                res = 1000 * Eigen::Vector2d::Ones();
            } else {
                res = _info_sqrt * (projection - _p2d);
            }
            return true;
        }

        // Factory to hide the construction of the CostFunction object from the client code.
        static ceres::CostFunction *Create(const Eigen::Vector2d &p2d,
                                           const std::shared_ptr<ImageSensor> &cam,
                                           const Eigen::Affine3d &T_w_lmk,
                                           const double sigma = 1.0) {
            return (new ceres::NumericDiffCostFunction<ReprojectionErrCeres_pointxd, ceres::FORWARD, 2, 6, 3>(
                new ReprojectionErrCeres_pointxd(p2d, cam, T_w_lmk, sigma)));
            // 6: dof first argument (framepose_vec), 3:dof second argument (landmark_p3d), 2: size of error
            // residual (residual)
        }
    };




    // For dealing with linexd landmarks
    struct ReprojectionErrCeres_linexd {
      public:

        ReprojectionErrCeres_linexd(const std::vector<Eigen::Vector2d> &p2ds,
                                    const std::shared_ptr<ImageSensor> &cam,
                                    const Eigen::Affine3d &T_w_lmk,
                                    const std::shared_ptr<AModel3d> &model3d,
                                    const double sigma = 1.0)
            : _p2ds(p2ds), _cam(cam), _T_w_lmk(T_w_lmk), _model3d(model3d) {
            _info_sqrt = (1 / sigma) * Eigen::Matrix2d::Identity();
        }

        // Constant parameters used to process the residual
        const std::vector<Eigen::Vector2d> _p2ds;
        const std::shared_ptr<ImageSensor> _cam;
        const Eigen::Affine3d _T_w_lmk;
        const std::shared_ptr<AModel3d> _model3d;
        Eigen::Matrix2d _info_sqrt;



        template <typename T> bool operator()(const T *const dX, const T *const dlmk, T *residual) const {

            // Get World to sensor transform
            Eigen::Affine3d T_f_w = _cam->getFrame()->getWorld2FrameTransform() * geometry::se3_doubleVec6dtoRT(dX);

            // Get Landmark P3Ds pose
            Eigen::Affine3d T_w_lmk = _T_w_lmk * geometry::se3_doubleVec3dtoRT(dlmk);

            std::vector<Eigen::Vector2d> projections;

            Eigen::Map<Eigen::Vector2d> res(residual);

            Eigen::Vector3d v_dir, v_proj1, v_proj2;

            if (!_cam->project(T_w_lmk, _model3d, T_f_w, projections)) {
                res = 1000 * Eigen::Vector2d::Ones();
            } else {

                // distance between projections and detected line "extremal" points (4 d residual...)
                // res << _info_sqrt * (projections.at(0) - _p2ds.at(0)), 
                       //_info_sqrt * (projections.at(1) - _p2ds.at(1));

                // distance from projections to 2D line
                v_dir << _p2ds.at(0)-_p2ds.at(1), 0.;
                v_proj1 << _p2ds.at(0)-projections.at(0), 0.;
                v_proj2 << _p2ds.at(0)-projections.at(1), 0.;

                res << _info_sqrt * Eigen::Vector2d((v_proj1.cross(v_dir)).norm()/v_dir.norm(), 
                                                    (v_proj2.cross(v_dir)).norm()/v_dir.norm());

            }
            
            return true;
        }

        // Factory to hide the construction of the CostFunction object from the client code.
        static ceres::CostFunction *Create(const std::vector<Eigen::Vector2d> &p2ds,
                                           const std::shared_ptr<ImageSensor> &cam,
                                           const Eigen::Affine3d &T_w_lmk,
                                           const std::shared_ptr<AModel3d> &model3d,
                                           const double sigma = 1.0) {
            return (new ceres::NumericDiffCostFunction<ReprojectionErrCeres_linexd, ceres::FORWARD, 2, 6, 6>(
                new ReprojectionErrCeres_linexd(p2ds, cam, T_w_lmk, model3d, sigma)));
            // 6: dof first argument (framepose_vec), 3:dof second argument (landmark_p3d), 2: size of error
            // residual (residual)
        }
    };

};

} // namespace isae

#endif // BUNDLEADJUSTMENTCERESNUMERIC_H
