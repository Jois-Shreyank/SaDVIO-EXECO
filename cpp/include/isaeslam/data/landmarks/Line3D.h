#ifndef LINE3D_H
#define LINE3D_H

#include "isaeslam/data/landmarks/ALandmark.h"

namespace isae {

/*!
 * @brief A 3D Line landmark class.
 * 
 * Line3D class represents a 3D line in space, labeled as "linexd".
 */
class Line3D : public ALandmark {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Line3D() {
        _label            = "linexd";
        _model            = std::make_shared<ModelLine3D>();
    }

    Line3D(std::vector<Eigen::Vector3d> T_w_l_vector, cv::Mat desc = cv::Mat()) : ALandmark() {
        _label            = "linexd";
        _model            = std::make_shared<ModelLine3D>();
    }

    Line3D(Eigen::Affine3d T_w_l, std::vector<std::shared_ptr<isae::AFeature>> features)
        : ALandmark(T_w_l, features) {
        _label            = "linexd";
        _model            = std::make_shared<ModelLine3D>();
    }

    double chi2err(std::shared_ptr<AFeature> f) override;


};

} // namespace isae

#endif // LINE3D_H
