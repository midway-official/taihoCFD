#pragma once

#include "solvers/flow_solver.h"
#include "numerics/momentum.h"

class SimpleSolver final : public FlowSolver {
public:
    explicit SimpleSolver(const SolverContext& context);

    SimpleSolver(
        Mesh& mesh,
        FluidProperties fluid,
        int rank,
        int num_procs,
        NumericalSchemes schemes,
        SolutionConfig solution);

    SolverIterationResult solveIteration(const TimeTerm& time_term) override;

private:
    SolverContext context_;
    Equation momentum_;
    Equation pressure_;
    Eigen::VectorXd source_v_;
    Eigen::VectorXd previous_u_;
    Eigen::VectorXd previous_v_;
};
