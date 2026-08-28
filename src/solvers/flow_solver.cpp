#include "solvers/flow_solver.h"

#include "solvers/simple_solver.h"

#include <mpi.h>

#include <stdexcept>

std::unique_ptr<FlowSolver> createFlowSolver(
    std::string_view algorithm,
    const SolverContext& context)
{
    context.validate();
    if (algorithm == "SIMPLE") {
        return std::make_unique<SimpleSolver>(context);
    }
    throw std::invalid_argument(
        "未注册的压力-速度耦合算法: " + std::string(algorithm));
}

std::unique_ptr<FlowSolver> createFlowSolver(
    std::string_view algorithm,
    Mesh& mesh,
    FluidProperties fluid,
    int rank,
    int size,
    NumericalSchemes schemes,
    SolutionConfig solution)
{
    SolverContext context{
        mesh, fluid, schemes, solution,
        ParallelContext{MPI_COMM_WORLD, rank, size}};
    return createFlowSolver(algorithm, context);
}
