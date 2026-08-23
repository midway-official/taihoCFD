#include "solvers/simple_solver.h"

#include "numerics/pressure_correction.h"
#include "numerics/rhie_chow.h"
#include "parallel/halo_exchange.h"

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
    Mesh& mesh,
    double viscosity,
    int rank,
    int num_procs,
    NumericalSchemes schemes,
    SolutionConfig solution)
    : mesh_(mesh),
      viscosity_(viscosity),
      rank_(rank),
      num_procs_(num_procs),
      schemes_(schemes),
      solution_(solution),
      momentum_(mesh),
      pressure_(mesh),
      source_v_(Eigen::VectorXd::Zero(mesh.internumber)),
      previous_u_(Eigen::VectorXd::Zero(mesh.internumber)),
      previous_v_(Eigen::VectorXd::Zero(mesh.internumber))
{
    if (!(viscosity_ > 0.0) || !std::isfinite(viscosity_)) {
        throw std::invalid_argument("动力粘度必须为正且有限");
    }
    if (rank_ < 0 || rank_ >= num_procs_) {
        throw std::invalid_argument("非法 MPI rank/size");
    }
    schemes_.validate();
    solution_.validate();
}

SimpleIterationResult SimpleSolver::solveIteration(const TimeTerm& time_term) {
    matrixToVector(mesh_.u_star, previous_u_, mesh_);
    matrixToVector(mesh_.v_star, previous_v_, mesh_);

    assembleMomentum(
        mesh_, momentum_, source_v_, viscosity_,
        solution_.simple.velocity_relaxation, time_term, schemes_);
    mesh_.u = mesh_.u_star;
    mesh_.v = mesh_.v_star;

    SimpleIterationResult result;
    result.u = solveField(
        momentum_, momentum_.source, mesh_, mesh_.u,
        solution_.velocity, rank_, num_procs_);
    result.v = solveField(
        momentum_, source_v_, mesh_, mesh_.v,
        solution_.velocity, rank_, num_procs_);

    exchangeColumns(momentum_.A_p, rank_, num_procs_);
    interpolateFaceVelocity(
        mesh_, momentum_, schemes_.face_interpolation);
    assemblePressureCorrection(
        mesh_, pressure_, momentum_, rank_, num_procs_);
    result.pressure = solveField(
        pressure_, pressure_.source, mesh_, mesh_.p_prime,
        solution_.pressure, rank_, num_procs_);

    correctPressure(mesh_, solution_.simple.pressure_relaxation);
    correctVelocity(mesh_, momentum_);
    exchangeColumns(mesh_.p, rank_, num_procs_);

    std::array<double, 6> local{};
    for (int n = 0; n < mesh_.internumber; ++n) {
        const int i = mesh_.interi[static_cast<std::size_t>(n)];
        const int j = mesh_.interj[static_cast<std::size_t>(n)];
        const double du = mesh_.u_star(i, j) - previous_u_[n];
        const double dv = mesh_.v_star(i, j) - previous_v_[n];
        local[0] += du * du;
        local[1] += dv * dv;
        local[2] += mesh_.u_star(i, j) * mesh_.u_star(i, j);
        local[3] += mesh_.v_star(i, j) * mesh_.v_star(i, j);
        local[4] += mesh_.p_prime(i, j) * mesh_.p_prime(i, j);
        local[5] += mesh_.p(i, j) * mesh_.p(i, j);
    }
    std::array<double, 6> global{};
    MPI_Allreduce(
        local.data(), global.data(), static_cast<int>(global.size()),
        MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    result.relative_velocity_change =
        std::sqrt(global[0] + global[1]) /
        std::max(std::sqrt(global[2] + global[3]), 1e-30);
    result.relative_pressure_correction =
        std::sqrt(global[4]) / std::max(std::sqrt(global[5]), 1e-30);
    result.continuity = computeContinuityMetrics(mesh_);
    result.healthy =
        isHealthy(result.u) && isHealthy(result.v) &&
        isHealthy(result.pressure) &&
        std::isfinite(result.relative_velocity_change) &&
        std::isfinite(result.continuity.relative);
    result.converged = result.healthy &&
        result.u.converged() && result.v.converged() &&
        result.pressure.converged() &&
        result.continuity.relative <= solution_.simple.residual.continuity &&
        result.relative_velocity_change <=
            solution_.simple.residual.velocity_change;
    return result;
}
