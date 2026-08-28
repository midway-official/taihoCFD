#pragma once

#include "numerics/fluid_properties.h"
#include "numerics/schemes.h"
#include "solvers/solver_config.h"

#include <filesystem>
#include <optional>
#include <string>

struct TimeControl {
    double delta_t;
    int steps;
};

struct CaseConfig {
    std::filesystem::path mesh_path;
    std::filesystem::path output_path;
    FluidProperties fluid;
    NumericalSchemes schemes;
    SolutionConfig solution;
    std::string algorithm;
    std::optional<TimeControl> time;

    bool transient() const {
        return schemes.time == TimeScheme::BackwardEuler;
    }

    void validate() const;
};

CaseConfig readCaseConfig(const std::filesystem::path& path);
