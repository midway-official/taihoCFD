#pragma once

#include "numerics/continuity.h"
#include "numerics/momentum.h"
#include "solvers/linear_solver.h"
#include "solvers/solution_config.h"

struct SimpleIterationResult {
    LinearSolverResult u;
    LinearSolverResult v;
    LinearSolverResult pressure;
    ContinuityMetrics continuity;
    double relative_velocity_change = 0.0;
    double relative_pressure_correction = 0.0;
    bool healthy = true;
    bool converged = false;
};

class SimpleSolver {
public:
    SimpleSolver(
        Mesh& mesh,
        double viscosity,
        int rank,
        int num_procs,
        NumericalSchemes schemes = NumericalSchemes::steady(),
        SolutionConfig solution = {});

    SimpleIterationResult solveIteration(const TimeTerm& time_term);

private:
    Mesh& mesh_;
    double viscosity_;
    int rank_;
    int num_procs_;
    NumericalSchemes schemes_;
    SolutionConfig solution_;
    Equation momentum_;
    Equation pressure_;
    Eigen::VectorXd source_v_;
    Eigen::VectorXd previous_u_;
    Eigen::VectorXd previous_v_;
};
