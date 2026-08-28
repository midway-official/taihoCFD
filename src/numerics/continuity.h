#pragma once

#include "mesh/mesh.h"
#include "parallel/parallel_context.h"

struct ContinuityMetrics {
    double l1 = 0.0;
    double l2 = 0.0;
    double max_abs = 0.0;
    double relative = 0.0;
};

ContinuityMetrics computeContinuityMetrics(
    const Mesh& mesh,
    const ParallelContext& parallel);
ContinuityMetrics computeContinuityMetrics(const Mesh& mesh);
