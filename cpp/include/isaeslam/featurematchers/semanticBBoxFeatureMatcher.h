#ifndef SEMANTICBBOXFEATUREMATCHER_H
#define SEMANTICBBOXFEATUREMATCHER_H

#include "isaeslam/typedefs.h"
#include <type_traits>

#include "isaeslam/featurematchers/afeaturematcher.h"

namespace isae {

/*!
 * @brief Class for matching 2D bouding box features
 */
class semanticBBoxFeatureMatcher : public AFeatureMatcher {
  public:
    semanticBBoxFeatureMatcher() {}
    semanticBBoxFeatureMatcher(std::shared_ptr<AFeatureDetector> detector) : AFeatureMatcher(detector) {
        _feature_label = "bboxxd";
    }
};

} // namespace isae

#endif // SEMANTICBBOXFEATUREMATCHER_H
