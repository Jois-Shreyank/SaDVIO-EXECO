#ifndef POINT3DLANDMARKINITIALIZER_H
#define POINT3DLANDMARKINITIALIZER_H

#include "isaeslam/data/landmarks/Point3D.h"
#include "isaeslam/landmarkinitializer/alandmarkinitializer.h"
#include "isaeslam/typedefs.h"

namespace isae {

/*!
 * @brief Class for initializing 3D point landmarks.
 */
class Point3DLandmarkInitializer : public ALandmarkInitializer {
  public:
    Point3DLandmarkInitializer() = default;

  private:
    bool initLandmark(std::vector<std::shared_ptr<AFeature>> features, std::shared_ptr<ALandmark> &landmark) override;
    bool initLandmarkWithDepth(std::vector<std::shared_ptr<AFeature>> features,
                               std::shared_ptr<ALandmark> &landmark) override;

    // must link f or f1 & f2 to the landmark
    std::shared_ptr<ALandmark> createNewLandmark(std::shared_ptr<AFeature> f) override;
};

} // namespace isae

#endif // POINT3DLANDMARKINITIALIZER_H
