#pragma once

#include "numerics/continuity.h"

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

struct SolverIterationResult {
    LinearSolverResult u;
    LinearSolverResult v;
    LinearSolverResult pressure;
    ContinuityMetrics continuity;
    double relative_velocity_change = 0.0;
    double relative_pressure_correction = 0.0;
    bool healthy = true;
    bool converged = false;
};

std::string_view toString(LinearSolverStatus status);
