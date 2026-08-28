#include "apps/case_config.h"
#include "io/mesh_reader.h"
#include "io/result_writer.h"
#include "mesh/boundary.h"
#include "parallel/domain_decomposition.h"
#include "solvers/flow_solver.h"

#include <mpi.h>

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void printSetup(
    const CaseConfig& config,
    const Mesh& mesh,
    const ParallelContext& parallel)
{
    int global_cells = 0;
    MPI_Reduce(
        &mesh.internumber, &global_cells, 1, MPI_INT, MPI_SUM, 0,
        parallel.communicator);
    if (parallel.rank != 0) {
        return;
    }
    std::cout << "TaihoCFD " << (config.transient() ? "transient" : "steady")
              << " | MPI=" << parallel.size << " | cells=" << global_cells
              << " | rho=" << config.fluid.rho << " | mu=" << config.fluid.mu;
    if (config.transient()) {
        std::cout << " | dt=" << config.time->delta_t
                  << " | steps=" << config.time->steps;
    }
    std::cout << "\nschemes: ddt=" << toString(config.schemes.time)
              << " div(U)=" << toString(config.schemes.velocity_convection)
              << " grad(p)=" << toString(config.schemes.pressure_gradient)
              << " laplacian(U)=" << toString(config.schemes.velocity_laplacian)
              << " interpolation=" << toString(config.schemes.face_interpolation)
              << "\nlinear: U=" << toString(config.solution.velocity.solver)
              << '/' << toString(config.solution.velocity.preconditioner)
              << " p=" << toString(config.solution.pressure.solver)
              << '/' << toString(config.solution.pressure.preconditioner)
              << " | SIMPLE max=" << config.solution.simple.max_iterations
              << " | mass tol=" << config.solution.simple.residual.continuity
              << " | dU tol=" << config.solution.simple.residual.velocity_change
              << '\n';
}

void printIteration(
    int iteration,
    const SolverIterationResult& result,
    const ParallelContext& parallel,
    const std::string& prefix)
{
    if (parallel.rank != 0) {
        return;
    }
    std::cout << prefix << "SIMPLE " << std::setw(4) << iteration
              << std::scientific << std::setprecision(3)
              << " | lin(u,v,p)=" << result.u.relative_residual << ','
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

void runCase(const CaseConfig& config, const ParallelContext& parallel) {
    Mesh mesh = extractLocalMesh(readMesh(config.mesh_path.string()), parallel);
    initializeFlowFields(mesh);
    printSetup(config, mesh, parallel);
    SolverContext context{
        mesh, config.fluid, config.schemes, config.solution, parallel};
    auto solver = createFlowSolver(config.algorithm, context);

    const int outer_steps = config.transient() ? config.time->steps : 1;
    const auto start = std::chrono::steady_clock::now();
    for (int step = 1; step <= outer_steps; ++step) {
        const TimeTerm time = config.transient()
            ? TimeTerm::backwardEuler(
                config.time->delta_t, mesh.u0, mesh.v0)
            : TimeTerm::none();
        bool converged = false;
        int completed = 0;
        for (int iteration = 1;
             iteration <= config.solution.simple.max_iterations; ++iteration) {
            const SolverIterationResult result = solver->solveIteration(time);
            completed = iteration;
            if (iteration == 1 || iteration % 10 == 0 ||
                result.converged || !result.healthy) {
                printIteration(
                    iteration, result, parallel,
                    config.transient() ? "time " + std::to_string(step) + " | " : "");
            }
            if (!result.healthy) {
                throw std::runtime_error("线性求解器发生数值失效");
            }
            if ((converged = result.converged)) {
                break;
            }
        }
        if (parallel.rank == 0 && (!config.transient() || !converged)) {
            std::cout << (config.transient() ? "time " + std::to_string(step) + " | " : "")
                      << (converged ? "SIMPLE converged" : "SIMPLE reached iteration limit")
                      << " | iterations=" << completed << '\n';
        }
        if (config.transient()) {
            mesh.u0 = mesh.u_star;
            mesh.v0 = mesh.v_star;
        }
    }

    saveMeshData(mesh, parallel.rank, config.output_path.string());
    if (parallel.rank == 0) {
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << "completed | elapsed=" << elapsed << " s\n";
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    const ParallelContext parallel = ParallelContext::world();
    try {
        if (argc != 1) {
            throw std::invalid_argument(
                "TaihoCFD 不接受命令行参数；请在含 case.cfg 的算例目录运行");
        }
        runCase(
            readCaseConfig(std::filesystem::current_path() / "case.cfg"),
            parallel);
    } catch (const std::exception& error) {
        std::cerr << "rank " << parallel.rank << " error: " << error.what()
                  << '\n';
        MPI_Abort(parallel.communicator, 2);
        return 2;
    }
    MPI_Finalize();
    return 0;
}
