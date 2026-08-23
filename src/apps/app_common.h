#pragma once

#include "solvers/simple_solver.h"

#include <string>

enum class SimulationMode {
    Steady,
    Unsteady,
};

struct SimulationParameters {
    std::string mesh_folder;
    double dt = 0.0;
    int steps = 0;
    double viscosity = 0.0;
};

SimulationParameters parseAndBroadcastParameters(
    SimulationMode mode,
    int argc,
    char* argv[],
    int rank,
    int num_procs);

void printSimulationSetup(
    SimulationMode mode,
    const SimulationParameters& parameters,
    const Mesh& local_mesh,
    int rank,
    int num_procs,
    const SolverConfig& config);

void printIterationResult(
    int iteration,
    const SimpleIterationResult& result,
    int rank,
    const char* prefix = "");

int runApplication(SimulationMode mode, int argc, char* argv[]);
