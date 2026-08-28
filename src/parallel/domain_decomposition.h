#pragma once

#include "mesh/mesh.h"
#include "parallel/parallel_context.h"

Mesh extractLocalMesh(const Mesh& original, const ParallelContext& parallel);
Mesh extractLocalMesh(const Mesh& original, int rank, int num_procs);
