#pragma once

#include <string_view>

enum class LinearSolverType {
    BiCGSTAB,
    PCG,
};

enum class PreconditionerType {
    ILUT,
    IncompleteCholesky,
};

struct LinearSolverConfig {
    LinearSolverType solver = LinearSolverType::BiCGSTAB;
    PreconditionerType preconditioner = PreconditionerType::ILUT;
    double absolute_tolerance = 1e-14;
    double relative_tolerance = 1e-7;
    int max_iterations = 200;
    bool warm_start = true;

    void validate() const;
};

struct ResidualControl {
    double continuity = 1e-7;
    double velocity_change = 1e-7;
};

struct SimpleControl {
    int max_iterations = 200;
    int non_orthogonal_correctors = 0;
    double pressure_relaxation = 0.3;
    double velocity_relaxation = 0.5;
    ResidualControl residual;

    void validate() const;
};

struct SolutionConfig {
    LinearSolverConfig velocity;
    LinearSolverConfig pressure{
        LinearSolverType::PCG,
        PreconditionerType::IncompleteCholesky,
        1e-14,
        1e-7,
        200,
        false,
    };
    SimpleControl simple;

    void validate() const;
};

std::string_view toString(LinearSolverType solver);
std::string_view toString(PreconditionerType preconditioner);
