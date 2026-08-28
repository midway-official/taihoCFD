#include "solvers/simple_solver.h"

#include "numerics/pressure_correction.h"
#include "numerics/rhie_chow.h"
#include "parallel/halo_exchange.h"
#include "solvers/linear_solver.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace {

bool isHealthy(const LinearSolverResult& result) {
    return result.status != LinearSolverStatus::Breakdown &&
        std::isfinite(result.final_residual) &&
        std::isfinite(result.relative_residual);
}

}  // namespace

SimpleSolver::SimpleSolver(
    const SolverContext& context)
    : context_(context),
      momentum_(context_.mesh),
      pressure_(context_.mesh),
      source_v_(Eigen::VectorXd::Zero(context_.mesh.internumber)),
      previous_u_(Eigen::VectorXd::Zero(context_.mesh.internumber)),
      previous_v_(Eigen::VectorXd::Zero(context_.mesh.internumber))
{
    context_.validate();
}

SimpleSolver::SimpleSolver(
    Mesh& mesh,
    FluidProperties fluid,
    int rank,
    int num_procs,
    NumericalSchemes schemes,
    SolutionConfig solution)
    : SimpleSolver(SolverContext{
          mesh, fluid, schemes,
          solution, ParallelContext{MPI_COMM_WORLD, rank, num_procs}})
{}

SolverIterationResult SimpleSolver::solveIteration(const TimeTerm& time_term) {
    Mesh& mesh = context_.mesh;
    const auto& parallel = context_.parallel;
    matrixToVector(mesh.u_star, previous_u_, mesh);
    matrixToVector(mesh.v_star, previous_v_, mesh);

    assembleMomentum(
        mesh, momentum_, source_v_, context_.fluid,
        context_.solution.simple.velocity_relaxation, time_term,
        context_.schemes);
    mesh.u = mesh.u_star;
    mesh.v = mesh.v_star;

    SolverIterationResult result;
    result.u = solveField(
        momentum_, momentum_.source, mesh, mesh.u,
        context_.solution.velocity, parallel);
    result.v = solveField(
        momentum_, source_v_, mesh, mesh.v,
        context_.solution.velocity, parallel);

    exchangeColumns(momentum_.A_p, parallel);
    interpolateFaceVelocity(
        mesh, momentum_, context_.schemes.face_interpolation);
    assemblePressureCorrection(
        mesh, pressure_, momentum_, parallel);
    result.pressure = solveField(
        pressure_, pressure_.source, mesh, mesh.p_prime,
        context_.solution.pressure, parallel);

    correctPressure(mesh, context_.solution.simple.pressure_relaxation);
    correctVelocity(mesh, momentum_);
    exchangeColumns(mesh.p, parallel);

    std::array<double, 6> local{};
    for (int n = 0; n < mesh.internumber; ++n) {
        const int i = mesh.interi[static_cast<std::size_t>(n)];
        const int j = mesh.interj[static_cast<std::size_t>(n)];
        const double du = mesh.u_star(i, j) - previous_u_[n];
        const double dv = mesh.v_star(i, j) - previous_v_[n];
        local[0] += du * du;
        local[1] += dv * dv;
        local[2] += mesh.u_star(i, j) * mesh.u_star(i, j);
        local[3] += mesh.v_star(i, j) * mesh.v_star(i, j);
        local[4] += mesh.p_prime(i, j) * mesh.p_prime(i, j);
        local[5] += mesh.p(i, j) * mesh.p(i, j);
    }
    std::array<double, 6> global{};
    MPI_Allreduce(
        local.data(), global.data(), static_cast<int>(global.size()),
        MPI_DOUBLE, MPI_SUM, parallel.communicator);

    result.relative_velocity_change =
        std::sqrt(global[0] + global[1]) /
        std::max(std::sqrt(global[2] + global[3]), 1e-30);
    result.relative_pressure_correction =
        std::sqrt(global[4]) / std::max(std::sqrt(global[5]), 1e-30);
    result.continuity = computeContinuityMetrics(mesh, parallel);
    result.healthy =
        isHealthy(result.u) && isHealthy(result.v) &&
        isHealthy(result.pressure) &&
        std::isfinite(result.relative_velocity_change) &&
        std::isfinite(result.continuity.relative);
    result.converged = result.healthy &&
        result.u.converged() && result.v.converged() &&
        result.pressure.converged() &&
        result.continuity.relative <=
            context_.solution.simple.residual.continuity &&
        result.relative_velocity_change <=
            context_.solution.simple.residual.velocity_change;
    return result;
}
