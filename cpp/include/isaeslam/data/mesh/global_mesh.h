#ifndef GLOBAL_MESH_H
#define GLOBAL_MESH_H

#include "isaeslam/data/mesh/mesh.h"

namespace isae {

class GlobalMesh {
  public:
    GlobalMesh() = default;
    ~GlobalMesh() = default;

    void addPolygons(const std::vector<std::shared_ptr<Polygon>>& new_polygons);
    std::vector<std::shared_ptr<Polygon>> getPolygonVector() const {
        std::lock_guard<std::mutex> lock(_mesh_mtx);
        return _polygons;
    }

  private:
    std::vector<std::shared_ptr<Polygon>> _polygons;
    mutable std::mutex _mesh_mtx;
};

} // namespace isae

#endif // GLOBAL_MESH_H