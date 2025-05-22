#ifndef AMAP_H
#define AMAP_H

#include <deque>

#include "isaeslam/data/frame.h"
#include "isaeslam/typedefs.h"

namespace isae {

class AMap {
  public:
    AMap() = default;

    // add a Frame
    virtual void addFrame(std::shared_ptr<isae::Frame> &frame) = 0;

    // getters
    std::deque<std::shared_ptr<Frame>> &getFrames() { return _frames; }

    std::shared_ptr<isae::Frame> getLastFrame() {
        if (_frames.empty())
            return nullptr;
        return _frames.back();
    }

    void getLastNFramesIn(size_t N, std::vector<std::shared_ptr<isae::Frame>> &dest) {
        for (uint i = 0; i < std::min(N, _frames.size()); ++i) {
            dest.push_back(_frames.at(_frames.size() - 1 - i));
        }
    }

    typed_vec_landmarks &getLandmarks() { return _landmarks; }
    size_t getMapSize() { return _frames.size(); }

    void pushLandmarks(std::shared_ptr<isae::Frame> &frame) {
        typed_vec_landmarks all_ldmks = frame->getLandmarks();

        // For all type of landmarks to add
        for (auto &typed_ldmks : all_ldmks) {
            for (auto &ldmk : typed_ldmks.second) {
                if (!(!ldmk->isInitialized() || ldmk->isInMap() || ldmk->getFeatures().empty())) {
                    {
                        ldmk->setInMap();
                        _landmarks[ldmk->getLandmarkLabel()].push_back(ldmk);
                    }
                }
            }
        }
    }

  protected:
    std::deque<std::shared_ptr<Frame>> _frames; // frames in the map
    typed_vec_landmarks _landmarks;             // all landmarks in the map
};

} // namespace isae

#endif // AMAP_H