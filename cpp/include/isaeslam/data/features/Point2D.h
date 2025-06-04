#ifndef POINT2D_H
#define POINT2D_H

#include "isaeslam/data/features/AFeature2D.h"

namespace isae {

/*! 
* @brief A single 2D point feature class.
*
* Point2D class represents a single 2D point feature in the image, labeled as "pointxd".
*/
class Point2D : public AFeature {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Point2D() : AFeature() { _feature_label = "pointxd"; }
    Point2D(std::vector<Eigen::Vector2d> poses2d, cv::Mat desc = cv::Mat(), int octave = 0, float response = 0)
        : AFeature(poses2d, desc, octave) {
        _feature_label = "pointxd";
        _response     = response;
    }
};

} // namespace isae

#endif // POINT2D_H
