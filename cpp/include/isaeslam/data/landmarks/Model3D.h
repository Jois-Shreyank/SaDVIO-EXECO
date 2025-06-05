#ifndef SLAM_ISAE_MODEL3D_H
#define SLAM_ISAE_MODEL3D_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <memory>

namespace isae {

/*!
* @brief Abstract class for 3D models.
*
* This class serves as a base for various 3D models used in SLAM, such as points, lines, bounding boxes etc...
*/
class AModel3d : public std::enable_shared_from_this<AModel3d> {
  public:
    AModel3d() {}
    std::vector<Eigen::Vector3d> getModel() { return model; }

  protected:
    std::vector<Eigen::Vector3d> model;
};

/*!
* @brief Model for a 3D point.
*/
class ModelPoint3D : public AModel3d {
  public:
    ModelPoint3D() { model.push_back(Eigen::Vector3d(0, 0, 0)); }
};


/*!
* @brief Model for a 3D Line
*/
class ModelLine3D : public AModel3d {
  public:
    ModelLine3D() {
        model.push_back(Eigen::Vector3d(-0.5, 0, 0)); // start point
        model.push_back(Eigen::Vector3d(0.5, 0, 0));  // end point
    }
};


/*!
* @brief Model for a 3D Bounding Box
*/
class ModelBBox3D : public AModel3d {
  public:
    ModelBBox3D() {
        model.push_back(Eigen::Vector3d(0, 0, 0));
        model.push_back(Eigen::Vector3d(0, 1, 0));
        model.push_back(Eigen::Vector3d(1, 0, 0));
        model.push_back(Eigen::Vector3d(1, 1, 0));
        model.push_back(Eigen::Vector3d(0, 0, 1));
        model.push_back(Eigen::Vector3d(0, 1, 1));
        model.push_back(Eigen::Vector3d(1, 0, 1));
        model.push_back(Eigen::Vector3d(1, 1, 1));
    }
};

} // namespace isae
#endif // SLAM_ISAE_MODEL3D_H
