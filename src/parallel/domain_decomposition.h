#pragma once

#include "mesh/mesh.h"

Mesh extractLocalMesh(const Mesh& original, int rank, int num_procs);
