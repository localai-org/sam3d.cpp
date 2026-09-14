// Minimal vertex-colour GLB export for the native SAM 3D Objects mesh.
#pragma once

#include "mesh_extractor.hpp"

#include <string>

namespace sam3d {

// Writes one indexed triangle primitive. Positions are converted from the
// upstream decoder's Z-up coordinates to glTF's Y-up coordinates, and the
// first three learned vertex-attribute channels are exported as RGB.
bool write_object_glb(const std::string& path, const ObjectMesh& mesh,
                      std::string* error = nullptr);

}  // namespace sam3d
