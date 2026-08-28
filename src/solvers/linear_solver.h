#pragma once

#include "numerics/equation.h"
#include "parallel/parallel_context.h"
#include "solvers/solver_config.h"
#include "solvers/solver_result.h"

#include <string_view>

LinearSolverResult solveField(
    const Equation& equation,
    const Eigen::VectorXd& right_hand_side,
    const Mesh& mesh,
    Eigen::MatrixXd& field,
    const LinearSolverConfig& config,
    const ParallelContext& parallel);

LinearSolverResult solveField(
    const Equation& equation,
    const Eigen::VectorXd& right_hand_side,
    const Mesh& mesh,
    Eigen::MatrixXd& field,
    const LinearSolverConfig& config,
    int rank,
    int num_procs);
