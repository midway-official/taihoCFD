#include "apps/app_common.h"

#include "io/mesh_reader.h"
#include "io/result_writer.h"
#include "mesh/boundary.h"
#include "parallel/domain_decomposition.h"

#include <mpi.h>

#include <chrono>
#include <iostream>
#include <stdexcept>

namespace {

SolverConfig solverConfig(SimulationMode mode) {
    SolverConfig config;
    config.pressure_relaxation = 0.3;
    config.velocity_relaxation =
        mode == SimulationMode::Steady ? 0.5 : 0.7;
    config.linear_tolerance = 1e-7;
    config.momentum_max_iterations = 200;
    config.pressure_max_iterations = 1000;
    config.continuity_tolerance = 1e-7;
    config.velocity_change_tolerance =
        mode == SimulationMode::Steady ? 1e-6 : 1e-4;
    return config;
}

void runSimulation(
    SimulationMode mode,
    const SimulationParameters& parameters,
    int rank,
    int num_procs)
{
    Mesh mesh = [&] {
        Mesh original = readMesh(parameters.mesh_folder);
        return extractLocalMesh(original, rank, num_procs);
    }();
    initializeFlowFields(mesh);

    const SolverConfig config = solverConfig(mode);
    printSimulationSetup(
        mode, parameters, mesh, rank, num_procs, config);
    SimpleSolver solver(
        mesh, parameters.viscosity, rank, num_procs, config);

    const int time_steps =
        mode == SimulationMode::Steady ? 1 : parameters.steps;
    const int simple_limit =
        mode == SimulationMode::Steady ? parameters.steps : 30;
    const auto start = std::chrono::steady_clock::now();

    for (int time_step = 1; time_step <= time_steps; ++time_step) {
        const TimeTerm time_term = mode == SimulationMode::Steady
            ? TimeTerm::none()
            : TimeTerm::backwardEuler(parameters.dt, mesh.u0, mesh.v0);
        bool converged = false;
        int completed_iterations = 0;

        for (int iteration = 1; iteration <= simple_limit; ++iteration) {
            const SimpleIterationResult result =
                solver.solveIteration(time_term);
            completed_iterations = iteration;
            if (iteration == 1 || iteration % 10 == 0 ||
                result.converged || !result.healthy) {
                const std::string prefix = mode == SimulationMode::Steady
                    ? std::string()
                    : "time " + std::to_string(time_step) + " | ";
                printIterationResult(
                    iteration, result, rank, prefix.c_str());
            }
            if (!result.healthy) {
                throw std::runtime_error("线性求解器发生数值失效");
            }
            if (result.converged) {
                converged = true;
                break;
            }
        }

        if (mode == SimulationMode::Steady) {
            if (rank == 0) {
                std::cout << (converged
                    ? "SIMPLE converged"
                    : "SIMPLE reached iteration limit")
                    << " | iterations=" << completed_iterations << '\n';
            }
        } else {
            if (!converged && rank == 0) {
                std::cout << "time " << time_step
                          << " | SIMPLE reached inner iteration limit\n";
            }
            mesh.u0 = mesh.u_star;
            mesh.v0 = mesh.v_star;
        }
    }

    saveMeshData(mesh, rank, "result");
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    if (rank == 0) {
        std::cout << (mode == SimulationMode::Steady
            ? "steady completed"
            : "unsteady completed")
            << " | elapsed=" << elapsed << " s\n";
    }
}

}  // namespace

int runApplication(SimulationMode mode, int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int num_procs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    try {
        const SimulationParameters parameters = parseAndBroadcastParameters(
            mode, argc, argv, rank, num_procs);
        runSimulation(mode, parameters, rank, num_procs);
    } catch (const std::exception& error) {
        std::cerr << "rank " << rank << " error: " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 2);
        return 2;
    }

    MPI_Finalize();
    return 0;
}
