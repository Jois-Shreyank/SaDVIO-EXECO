#ifndef SEMANTICBBOXLANDMARKINITIALIZER_H
#define SEMANTICBBOXLANDMARKINITIALIZER_H

#include "isaeslam/data/landmarks/BBox3d.h"
#include "isaeslam/landmarkinitializer/alandmarkinitializer.h"
#include "isaeslam/typedefs.h"

namespace isae {

/*!
 * @brief Class for initializing semantic bounding box landmarks.
 */
class semanticBBoxLandmarkInitializer : public ALandmarkInitializer {
  public:
    semanticBBoxLandmarkInitializer() = default;

  private:
    bool initLandmark(std::vector<std::shared_ptr<AFeature>> features, std::shared_ptr<ALandmark> &landmark) override;
    bool initLandmarkWithDepth(std::vector<std::shared_ptr<AFeature>> features,
                               std::shared_ptr<ALandmark> &landmark) override;

    // must link f or f1 & f2 to the landmark
    std::shared_ptr<ALandmark> createNewLandmark(std::shared_ptr<AFeature> f) override;
};

} // namespace isae

#endif // SEMANTICBBOXLANDMARKINITIALIZER_H
