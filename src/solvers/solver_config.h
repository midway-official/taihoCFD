#pragma once

#include <optional>
#include <string_view>

enum class LinearSolverType {
    Unset,
    BiCGSTAB,
    PCG,
};

enum class PreconditionerType {
    Unset,
    ILUT,
    IncompleteCholesky,
};

struct LinearSolverConfig {
    LinearSolverType solver = LinearSolverType::Unset;
    PreconditionerType preconditioner = PreconditionerType::Unset;
    double absolute_tolerance = 0.0;
    double relative_tolerance = 0.0;
    int max_iterations = 0;
    std::optional<bool> warm_start;

    void validate() const;
};

struct ResidualControl {
    double continuity = 0.0;
    double velocity_change = 0.0;
};

struct SimpleControl {
    int max_iterations = 0;
    double pressure_relaxation = 0.0;
    double velocity_relaxation = 0.0;
    ResidualControl residual;

    void validate() const;
};

struct SolutionConfig {
    LinearSolverConfig velocity;
    LinearSolverConfig pressure;
    SimpleControl simple;

    void validate() const;
};

std::string_view toString(LinearSolverType solver);
std::string_view toString(PreconditionerType preconditioner);
