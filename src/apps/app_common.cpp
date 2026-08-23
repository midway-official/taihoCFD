#include "apps/app_common.h"

#include <mpi.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace {

int parsePositiveInt(const char* text, const char* label) {
    std::size_t used = 0;
    const std::string value(text);
    const int parsed = std::stoi(value, &used);
    if (used != value.size() || parsed <= 0) {
        throw std::invalid_argument(std::string(label) + " 必须为正整数");
    }
    return parsed;
}

double parsePositiveDouble(const char* text, const char* label) {
    std::size_t used = 0;
    const std::string value(text);
    const double parsed = std::stod(value, &used);
    if (used != value.size() || !(parsed > 0.0) || !std::isfinite(parsed)) {
        throw std::invalid_argument(std::string(label) + " 必须为正有限数");
    }
    return parsed;
}

void broadcastString(std::string& value, int rank) {
    int length = rank == 0 ? static_cast<int>(value.size()) : 0;
    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (length <= 0) {
        throw std::runtime_error("网格目录不能为空");
    }
    if (rank != 0) {
        value.resize(static_cast<std::size_t>(length));
    }
    MPI_Bcast(value.data(), length, MPI_CHAR, 0, MPI_COMM_WORLD);
}

}  // namespace

SimulationParameters parseAndBroadcastParameters(
    SimulationMode mode,
    int argc,
    char* argv[],
    int rank,
    int num_procs)
{
    SimulationParameters parameters;
    if (rank == 0) {
        const int expected = mode == SimulationMode::Steady ? 4 : 5;
        if (argc != expected) {
            const char* usage = mode == SimulationMode::Steady
                ? "用法: solver_simple_steady <mesh_folder> <max_simple_iterations> <viscosity>"
                : "用法: solver_simple_unsteady <mesh_folder> <dt> <time_steps> <viscosity>";
            throw std::invalid_argument(usage);
        }
        parameters.mesh_folder = argv[1];
        if (mode == SimulationMode::Steady) {
            parameters.steps = parsePositiveInt(argv[2], "最大 SIMPLE 迭代数");
            parameters.viscosity = parsePositiveDouble(argv[3], "动力粘度");
        } else {
            parameters.dt = parsePositiveDouble(argv[2], "时间步长");
            parameters.steps = parsePositiveInt(argv[3], "时间步数");
            parameters.viscosity = parsePositiveDouble(argv[4], "动力粘度");
        }
    }

    broadcastString(parameters.mesh_folder, rank);
    MPI_Bcast(&parameters.dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&parameters.steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&parameters.viscosity, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (num_procs <= 0 || parameters.steps <= 0 ||
        !(parameters.viscosity > 0.0) ||
        (mode == SimulationMode::Unsteady && !(parameters.dt > 0.0))) {
        throw std::runtime_error("广播后的求解参数无效");
    }
    return parameters;
}

void printSimulationSetup(
    SimulationMode mode,
    const SimulationParameters& parameters,
    const Mesh& local_mesh,
    int rank,
    int num_procs,
    const NumericalSchemes& schemes,
    const SolutionConfig& solution)
{
    int local_cells = local_mesh.internumber;
    int global_cells = 0;
    MPI_Reduce(
        &local_cells, &global_cells, 1, MPI_INT, MPI_SUM, 0,
        MPI_COMM_WORLD);
    if (rank != 0) {
        return;
    }

    std::cout << "TaihoCFD "
              << (mode == SimulationMode::Steady ? "steady" : "unsteady")
              << " | MPI=" << num_procs
              << " | internal cells=" << global_cells
              << " | mu=" << parameters.viscosity;
    if (mode == SimulationMode::Unsteady) {
        std::cout << " | dt=" << parameters.dt
                  << " | time steps=" << parameters.steps;
    } else {
        std::cout << " | max SIMPLE=" << parameters.steps;
    }
    std::cout << "\nschemes: ddt=" << toString(schemes.time)
              << " div(U)=" << toString(schemes.velocity_convection)
              << " grad(p)=" << toString(schemes.pressure_gradient)
              << " laplacian(U)=" << toString(schemes.velocity_laplacian)
              << " interpolation=" << toString(schemes.face_interpolation)
              << "\nlinear: U=" << toString(solution.velocity.solver)
              << '/' << toString(solution.velocity.preconditioner)
              << " p=" << toString(solution.pressure.solver)
              << '/' << toString(solution.pressure.preconditioner)
              << " | relTol(U,p)="
              << solution.velocity.relative_tolerance << ','
              << solution.pressure.relative_tolerance
              << " | continuity tol=" << solution.simple.residual.continuity
              << " | velocity-change tol="
              << solution.simple.residual.velocity_change << '\n';
}

void printIterationResult(
    int iteration,
    const SimpleIterationResult& result,
    int rank,
    const char* prefix)
{
    if (rank != 0) {
        return;
    }
    std::cout << prefix << "SIMPLE " << std::setw(4) << iteration
              << std::scientific << std::setprecision(3)
              << " | lin(u,v,p)="
              << result.u.relative_residual << ','
              << result.v.relative_residual << ','
              << result.pressure.relative_residual
              << " | mass=" << result.continuity.relative
              << " | dU=" << result.relative_velocity_change;
    if (!result.healthy) {
        std::cout << " | solver breakdown";
    } else if (result.u.status == LinearSolverStatus::MaxIterations ||
               result.v.status == LinearSolverStatus::MaxIterations ||
               result.pressure.status == LinearSolverStatus::MaxIterations) {
        std::cout << " | linear max-iter";
    }
    std::cout << '\n';
}
