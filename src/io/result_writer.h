#pragma once

#include "mesh/mesh.h"

#include <string>

void saveMeshData(
    const Mesh& mesh,
    int rank,
    const std::string& output_folder,
    bool owned_only = true);
