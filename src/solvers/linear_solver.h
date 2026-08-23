#pragma once

#include "numerics/equation.h"

#include <string_view>

enum class LinearSolverStatus {
    Converged,
    MaxIterations,
    Breakdown,
};

struct LinearSolverResult {
    LinearSolverStatus status = LinearSolverStatus::MaxIterations;
    int iterations = 0;
    double initial_residual = 0.0;
    double final_residual = 0.0;
    double relative_residual = 0.0;

    bool converged() const { return status == LinearSolverStatus::Converged; }
};

std::string_view toString(LinearSolverStatus status);

LinearSolverResult solveFieldBiCGSTAB(
    const Equation& equation,
    const Eigen::VectorXd& right_hand_side,
    const Mesh& mesh,
    Eigen::MatrixXd& field,
    double tolerance,
    int max_iterations,
    int rank,
    int num_procs,
    bool warm_start = true);

LinearSolverResult solveFieldPCG(
    const Equation& equation,
    const Eigen::VectorXd& right_hand_side,
    const Mesh& mesh,
    Eigen::MatrixXd& field,
    double tolerance,
    int max_iterations,
    int rank,
    int num_procs,
    bool warm_start = false);
