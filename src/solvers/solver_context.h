#pragma once

#include "mesh/mesh.h"
#include "numerics/fluid_properties.h"
#include "numerics/schemes.h"
#include "parallel/parallel_context.h"
#include "solvers/solver_config.h"

struct SolverContext {
    Mesh& mesh;
    FluidProperties fluid;
    NumericalSchemes schemes;
    SolutionConfig solution;
    ParallelContext parallel;

    void validate() const {
        parallel.validate();
        fluid.validate();
        schemes.validate();
        solution.validate();
    }
};
