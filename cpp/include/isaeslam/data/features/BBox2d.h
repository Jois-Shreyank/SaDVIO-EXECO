#ifndef BBOX2D_H
#define BBOX2D_H

#include "isaeslam/data/features/AFeature2D.h"

namespace isae {

/*!
 * @brief A 2D bounding box feature class.
 * 
 * BBox2D class represents a bounding box feature in the image, labeled as "bboxxd".
 * It inherits from AFeature and can hold multiple 2D points representing the corners of the bounding box.
 */
class BBox2D : public AFeature {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    BBox2D() { _feature_label = "bboxxd"; }
    BBox2D(std::vector<Eigen::Vector2d> poses2d, cv::Mat desc = cv::Mat()) : AFeature(poses2d, desc) {
        _feature_label = "bboxxd";
    }
};

} // namespace isae

#endif // BBOX2D_H
