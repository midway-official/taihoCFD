#pragma once

#include "solvers/solver_context.h"
#include "solvers/solver_result.h"
#include "numerics/time_term.h"

#include <memory>
#include <string_view>

class FlowSolver {
public:
    virtual ~FlowSolver() = default;
    virtual SolverIterationResult solveIteration(const TimeTerm& time) = 0;
};

std::unique_ptr<FlowSolver> createFlowSolver(
    std::string_view algorithm,
    const SolverContext& context);

std::unique_ptr<FlowSolver> createFlowSolver(
    std::string_view algorithm,
    Mesh& mesh,
    FluidProperties fluid,
    int rank,
    int size,
    NumericalSchemes schemes,
    SolutionConfig solution);
