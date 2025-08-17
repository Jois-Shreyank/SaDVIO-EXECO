#include "isaeslam/data/mesh/global_mesh.h"

namespace isae {

void GlobalMesh::addPolygons(const std::vector<std::shared_ptr<Polygon>>& new_polygons) {
    std::lock_guard<std::mutex> lock(_mesh_mtx);
    for (const auto& new_poly : new_polygons) {
        // Simple check to avoid duplicates, you might want a more robust method
        bool found = false;
        for (const auto& existing_poly : _polygons) {
            if (new_poly == existing_poly) {
                found = true;
                break;
            }
        }
        if (!found) {
            _polygons.push_back(new_poly);
        }
    }
}

} // namespace isae