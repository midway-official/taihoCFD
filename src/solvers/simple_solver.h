#pragma once

#include "numerics/continuity.h"
#include "numerics/momentum.h"
#include "solvers/linear_solver.h"

struct SolverConfig {
    double pressure_relaxation = 0.3;
    double velocity_relaxation = 0.5;
    double linear_tolerance = 1e-7;
    int momentum_max_iterations = 200;
    int pressure_max_iterations = 200;
    double continuity_tolerance = 1e-7;
    double velocity_change_tolerance = 1e-7;
};

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
        SolverConfig config = {});

    SimpleIterationResult solveIteration(const TimeTerm& time_term);

private:
    Mesh& mesh_;
    double viscosity_;
    int rank_;
    int num_procs_;
    SolverConfig config_;
    Equation momentum_;
    Equation pressure_;
    Eigen::VectorXd source_v_;
    Eigen::VectorXd previous_u_;
    Eigen::VectorXd previous_v_;
};
