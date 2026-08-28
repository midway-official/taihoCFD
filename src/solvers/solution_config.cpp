#include "solvers/solver_config.h"

#include <cmath>
#include <stdexcept>

namespace {

bool validTolerance(double value) {
    return value > 0.0 && std::isfinite(value);
}

}  // namespace

void LinearSolverConfig::validate() const {
    if (!validTolerance(absolute_tolerance) ||
        !validTolerance(relative_tolerance) || max_iterations <= 0 ||
        !warm_start.has_value()) {
        throw std::invalid_argument(
            "线性求解器容差、迭代上限或 warm_start 无效");
    }
    const bool valid_pair =
        (solver == LinearSolverType::BiCGSTAB &&
         preconditioner == PreconditionerType::ILUT) ||
        (solver == LinearSolverType::PCG &&
         preconditioner == PreconditionerType::IncompleteCholesky);
    if (!valid_pair) {
        throw std::invalid_argument("当前实现不支持该求解器/预条件组合");
    }
}

void SimpleControl::validate() const {
    if (max_iterations <= 0 ||
        !(pressure_relaxation > 0.0 && pressure_relaxation <= 1.0) ||
        !(velocity_relaxation > 0.0 && velocity_relaxation <= 1.0) ||
        !validTolerance(residual.continuity) ||
        !validTolerance(residual.velocity_change)) {
        throw std::invalid_argument("SIMPLE 控制参数无效");
    }
}

void SolutionConfig::validate() const {
    velocity.validate();
    pressure.validate();
    simple.validate();
}

std::string_view toString(LinearSolverType solver) {
    switch (solver) {
        case LinearSolverType::Unset:
            return "unset";
        case LinearSolverType::BiCGSTAB:
            return "BiCGSTAB";
        case LinearSolverType::PCG:
            return "PCG";
    }
    return "unknown";
}

std::string_view toString(PreconditionerType preconditioner) {
    switch (preconditioner) {
        case PreconditionerType::Unset:
            return "unset";
        case PreconditionerType::ILUT:
            return "ILUT";
        case PreconditionerType::IncompleteCholesky:
            return "incompleteCholesky";
    }
    return "unknown";
}
