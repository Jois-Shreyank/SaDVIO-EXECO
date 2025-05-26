#include "isaeslam/data/maps/globalmap.h"

namespace isae {

void GlobalMap::addFrame(std::shared_ptr<isae::Frame> &frame) {
    // A KF has been voted, the frame is added to the local map
    _frames.push_back(frame);

    // Add landmarks to the map
    this->pushLandmarks(frame);
}

} // namespace isae