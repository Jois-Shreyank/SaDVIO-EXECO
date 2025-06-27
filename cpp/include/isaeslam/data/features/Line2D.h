#ifndef LINE2D_H
#define LINE2D_H

#include "isaeslam/data/features/AFeature2D.h"

namespace isae {

/*! 
 * @brief A 2D line feature class.
 *
 * Line2D class represents a line feature in the image, labeled as "linexd".
 * It inherits from AFeature and can hold multiple 2D points representing the line.
 */
class Line2D : public AFeature {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Line2D() { _feature_label = "linexd"; }
    Line2D(std::vector<Eigen::Vector2d> poses2d, cv::Mat desc = cv::Mat()) : AFeature(poses2d, desc) {
        _feature_label = "linexd";
    }
};

} // namespace isae

#endif // LINE2D_H
