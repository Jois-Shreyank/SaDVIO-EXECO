#include "isaeslam/data/maps/globalmap.h"
#include <iostream>

namespace isae {

void GlobalMap::addFrame(std::shared_ptr<isae::Frame> &frame) {
    _frames.push_back(frame);

    typed_vec_landmarks all_ldmks = frame->getLandmarks();
    size_t inserted = 0;

    for (auto &typed_ldmks : all_ldmks) {
        for (auto &ldmk : typed_ldmks.second) {
            // Keep only basic sanity checks: initialized + has features
            if (!ldmk->isInitialized() || ldmk->getFeatures().empty())
                continue;

            // Don’t call ldmk->setInMap() here, otherwise you’ll block reuse later
            _landmarks[typed_ldmks.first].push_back(ldmk);
            inserted++;
        }
    }

    // Console debug (std::cout, not ROS)
    auto it = _landmarks.find("pointxd");
    const size_t total_points = (it == _landmarks.end()) ? 0 : it->second.size();

    std::cout << "GlobalMap: inserted " << inserted
              << " landmarks from frame; total point landmarks now = "
              << total_points << std::endl;
}

} // namespace isae