#pragma once

#include "mesh/mesh.h"

struct ContinuityMetrics {
    double l1 = 0.0;
    double l2 = 0.0;
    double max_abs = 0.0;
    double relative = 0.0;
};

ContinuityMetrics computeContinuityMetrics(const Mesh& mesh);
