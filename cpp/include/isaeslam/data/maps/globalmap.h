#ifndef GLOBALMAP_H
#define GLOBALMAP_H

#include "isaeslam/data/maps/amap.h"

namespace isae {

/*!
 * @brief Class for a Global Map
 *
 * This is a simple map with no particular strategy
 */
class GlobalMap : public AMap {
  public:
    GlobalMap() = default;

    /*!
     * @brief Add a frame to the global map.
     *
     * This method adds a frame to the global map and also pushes its landmarks into the map.
     * It is a simple implementation that does not enforce any sliding window or other constraints.
     */
    void addFrame(std::shared_ptr<Frame> &frame) override;

};

} // namespace isae

#endif // GLOBALMAP_H