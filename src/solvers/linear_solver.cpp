#include "solvers/linear_solver.h"

#include "parallel/halo_exchange.h"

#include <eigen3/Eigen/IterativeLinearSolvers>
#include <eigen3/unsupported/Eigen/IterativeSolvers>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

constexpr double breakdown_tolerance = 1e-30;
constexpr double absolute_residual_tolerance = 1e-14;

double globalDot(const Eigen::VectorXd& left, const Eigen::VectorXd& right) {
    const double local = left.dot(right);
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global;
}

double globalNorm(const Eigen::VectorXd& value) {
    return std::sqrt(std::max(globalDot(value, value), 0.0));
}

class DistributedOperator {
public:
    DistributedOperator(
        const Equation& equation,
        const Mesh& mesh,
        int rank,
        int num_procs)
        : equation_(equation),
          mesh_(mesh),
          rank_(rank),
          num_procs_(num_procs),
          exchange_field_(Eigen::MatrixXd::Zero(mesh.ny, mesh.nx))
    {}

    void apply(const Eigen::VectorXd& input, Eigen::VectorXd& output) {
        output.noalias() = equation_.A * input;
        vectorToMatrix(input, exchange_field_, mesh_);
        exchangeColumns(exchange_field_, rank_, num_procs_);

        for (int n = 0; n < mesh_.internumber; ++n) {
            const int i = mesh_.interi[static_cast<std::size_t>(n)];
            const int j = mesh_.interj[static_cast<std::size_t>(n)];
            if (isMpiGhost(mesh_.bctype(i, j + 1))) {
                output[n] -= equation_.A_e(i, j) * exchange_field_(i, j + 1);
            }
            if (isMpiGhost(mesh_.bctype(i, j - 1))) {
                output[n] -= equation_.A_w(i, j) * exchange_field_(i, j - 1);
            }
        }
    }

private:
    const Equation& equation_;
    const Mesh& mesh_;
    int rank_;
    int num_procs_;
    Eigen::MatrixXd exchange_field_;
};

double convergenceScale(double initial_residual, double right_hand_side_norm) {
    return std::max({initial_residual, right_hand_side_norm, 1e-30});
}

LinearSolverResult finalResult(
    LinearSolverStatus status,
    int iterations,
    double initial_residual,
    double tolerance,
    const Eigen::VectorXd& right_hand_side,
    const Eigen::VectorXd& solution,
    DistributedOperator& distributed_operator)
{
    Eigen::VectorXd matrix_solution(solution.size());
    distributed_operator.apply(solution, matrix_solution);
    const double final_residual = globalNorm(right_hand_side - matrix_solution);
    const double right_hand_side_norm = globalNorm(right_hand_side);
    const double reporting_scale = std::max(
        convergenceScale(initial_residual, right_hand_side_norm),
        absolute_residual_tolerance / tolerance);

    return {
        status,
        iterations,
        initial_residual,
        final_residual,
        final_residual / reporting_scale,
    };
}

bool invalidScalar(double value) {
    return !std::isfinite(value);
}

}  // namespace

std::string_view toString(LinearSolverStatus status) {
    switch (status) {
        case LinearSolverStatus::Converged:
            return "converged";
        case LinearSolverStatus::MaxIterations:
            return "max-iterations";
        case LinearSolverStatus::Breakdown:
            return "breakdown";
    }
    return "unknown";
}

LinearSolverResult solveFieldPCG(
    const Equation& equation,
    const Eigen::VectorXd& right_hand_side,
    const Mesh& mesh,
    Eigen::MatrixXd& field,
    double tolerance,
    int max_iterations,
    int rank,
    int num_procs,
    bool warm_start)
{
    if (!(tolerance > 0.0) || max_iterations <= 0 ||
        right_hand_side.size() != mesh.internumber) {
        throw std::invalid_argument("PCG 参数无效");
    }

    Eigen::VectorXd solution(mesh.internumber);
    if (warm_start) {
        matrixToVector(field, solution, mesh);
    } else {
        solution.setZero();
    }

    DistributedOperator distributed_operator(equation, mesh, rank, num_procs);
    Eigen::VectorXd matrix_solution(mesh.internumber);
    distributed_operator.apply(solution, matrix_solution);
    Eigen::VectorXd residual = right_hand_side - matrix_solution;

    Eigen::IncompleteCholesky<double> preconditioner;
    preconditioner.compute(equation.A);
    if (preconditioner.info() != Eigen::Success) {
        throw std::runtime_error("压力矩阵的 incomplete-Cholesky 分解失败");
    }
    Eigen::VectorXd preconditioned = preconditioner.solve(residual);
    Eigen::VectorXd direction = preconditioned;
    Eigen::VectorXd matrix_direction(mesh.internumber);

    const double initial_residual = globalNorm(residual);
    const double scale = convergenceScale(
        initial_residual, globalNorm(right_hand_side));
    const double target = std::max(
        tolerance * scale, absolute_residual_tolerance);
    if (initial_residual <= target) {
        vectorToMatrix(solution, field, mesh);
        exchangeColumns(field, rank, num_procs);
        return finalResult(
            LinearSolverStatus::Converged, 0, initial_residual, tolerance,
            right_hand_side, solution, distributed_operator);
    }

    double residual_preconditioned = globalDot(residual, preconditioned);
    LinearSolverStatus status = LinearSolverStatus::MaxIterations;
    int iterations = 0;

    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
        distributed_operator.apply(direction, matrix_direction);
        const double direction_matrix_direction =
            globalDot(direction, matrix_direction);
        if (invalidScalar(direction_matrix_direction) ||
            direction_matrix_direction <= breakdown_tolerance ||
            invalidScalar(residual_preconditioned)) {
            status = LinearSolverStatus::Breakdown;
            iterations = iteration - 1;
            break;
        }

        const double alpha = residual_preconditioned /
            direction_matrix_direction;
        solution.noalias() += alpha * direction;
        residual.noalias() -= alpha * matrix_direction;
        const double residual_norm = globalNorm(residual);
        iterations = iteration;
        if (invalidScalar(residual_norm)) {
            status = LinearSolverStatus::Breakdown;
            break;
        }
        if (residual_norm <= target) {
            status = LinearSolverStatus::Converged;
            break;
        }

        preconditioned = preconditioner.solve(residual);
        if (preconditioner.info() != Eigen::Success) {
            status = LinearSolverStatus::Breakdown;
            break;
        }
        const double next_residual_preconditioned =
            globalDot(residual, preconditioned);
        if (invalidScalar(next_residual_preconditioned) ||
            std::abs(residual_preconditioned) <= breakdown_tolerance) {
            status = LinearSolverStatus::Breakdown;
            break;
        }
        const double beta = next_residual_preconditioned /
            residual_preconditioned;
        direction = preconditioned + beta * direction;
        residual_preconditioned = next_residual_preconditioned;
    }

    vectorToMatrix(solution, field, mesh);
    exchangeColumns(field, rank, num_procs);
    return finalResult(
        status, iterations, initial_residual, tolerance,
        right_hand_side, solution, distributed_operator);
}

LinearSolverResult solveFieldBiCGSTAB(
    const Equation& equation,
    const Eigen::VectorXd& right_hand_side,
    const Mesh& mesh,
    Eigen::MatrixXd& field,
    double tolerance,
    int max_iterations,
    int rank,
    int num_procs,
    bool warm_start)
{
    if (!(tolerance > 0.0) || max_iterations <= 0 ||
        right_hand_side.size() != mesh.internumber) {
        throw std::invalid_argument("BiCGSTAB 参数无效");
    }

    Eigen::VectorXd solution(mesh.internumber);
    if (warm_start) {
        matrixToVector(field, solution, mesh);
    } else {
        solution.setZero();
    }

    DistributedOperator distributed_operator(equation, mesh, rank, num_procs);
    Eigen::VectorXd matrix_solution(mesh.internumber);
    distributed_operator.apply(solution, matrix_solution);
    Eigen::VectorXd residual = right_hand_side - matrix_solution;
    const Eigen::VectorXd shadow_residual = residual;

    Eigen::IncompleteLUT<double> preconditioner;
    preconditioner.setDroptol(1e-3);
    preconditioner.setFillfactor(2);
    preconditioner.compute(equation.A);
    if (preconditioner.info() != Eigen::Success) {
        throw std::runtime_error("动量矩阵的 ILUT 分解失败");
    }

    Eigen::VectorXd direction = Eigen::VectorXd::Zero(mesh.internumber);
    Eigen::VectorXd matrix_direction = Eigen::VectorXd::Zero(mesh.internumber);
    Eigen::VectorXd preconditioned_direction(mesh.internumber);
    Eigen::VectorXd intermediate(mesh.internumber);
    Eigen::VectorXd preconditioned_intermediate(mesh.internumber);
    Eigen::VectorXd matrix_intermediate(mesh.internumber);

    const double initial_residual = globalNorm(residual);
    const double scale = convergenceScale(
        initial_residual, globalNorm(right_hand_side));
    const double target = std::max(
        tolerance * scale, absolute_residual_tolerance);
    if (initial_residual <= target) {
        vectorToMatrix(solution, field, mesh);
        exchangeColumns(field, rank, num_procs);
        return finalResult(
            LinearSolverStatus::Converged, 0, initial_residual, tolerance,
            right_hand_side, solution, distributed_operator);
    }

    double previous_rho = 1.0;
    double alpha = 1.0;
    double omega = 1.0;
    LinearSolverStatus status = LinearSolverStatus::MaxIterations;
    int iterations = 0;

    for (int iteration = 1; iteration <= max_iterations; ++iteration) {
        const double rho = globalDot(shadow_residual, residual);
        if (invalidScalar(rho) || std::abs(rho) <= breakdown_tolerance ||
            invalidScalar(omega) || std::abs(omega) <= breakdown_tolerance) {
            status = LinearSolverStatus::Breakdown;
            iterations = iteration - 1;
            break;
        }

        const double beta = (rho / previous_rho) * (alpha / omega);
        direction = residual + beta * (direction - omega * matrix_direction);
        preconditioned_direction = preconditioner.solve(direction);
        if (preconditioner.info() != Eigen::Success) {
            status = LinearSolverStatus::Breakdown;
            iterations = iteration - 1;
            break;
        }
        distributed_operator.apply(preconditioned_direction, matrix_direction);

        const double shadow_matrix_direction =
            globalDot(shadow_residual, matrix_direction);
        if (invalidScalar(shadow_matrix_direction) ||
            std::abs(shadow_matrix_direction) <= breakdown_tolerance) {
            status = LinearSolverStatus::Breakdown;
            iterations = iteration - 1;
            break;
        }
        alpha = rho / shadow_matrix_direction;
        intermediate = residual - alpha * matrix_direction;

        const double intermediate_norm = globalNorm(intermediate);
        if (invalidScalar(intermediate_norm)) {
            status = LinearSolverStatus::Breakdown;
            iterations = iteration;
            break;
        }
        if (intermediate_norm <= target) {
            solution.noalias() += alpha * preconditioned_direction;
            residual = intermediate;
            status = LinearSolverStatus::Converged;
            iterations = iteration;
            break;
        }

        preconditioned_intermediate = preconditioner.solve(intermediate);
        if (preconditioner.info() != Eigen::Success) {
            status = LinearSolverStatus::Breakdown;
            iterations = iteration;
            break;
        }
        distributed_operator.apply(
            preconditioned_intermediate, matrix_intermediate);

        const double local_products[2] = {
            matrix_intermediate.dot(intermediate),
            matrix_intermediate.squaredNorm(),
        };
        double global_products[2] = {0.0, 0.0};
        MPI_Allreduce(
            local_products, global_products, 2, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
        if (invalidScalar(global_products[0]) ||
            invalidScalar(global_products[1]) ||
            global_products[1] <= breakdown_tolerance) {
            status = LinearSolverStatus::Breakdown;
            iterations = iteration;
            break;
        }
        omega = global_products[0] / global_products[1];
        if (invalidScalar(omega) || std::abs(omega) <= breakdown_tolerance) {
            status = LinearSolverStatus::Breakdown;
            iterations = iteration;
            break;
        }

        solution.noalias() +=
            alpha * preconditioned_direction +
            omega * preconditioned_intermediate;
        residual = intermediate - omega * matrix_intermediate;
        const double residual_norm = globalNorm(residual);
        iterations = iteration;
        if (invalidScalar(residual_norm)) {
            status = LinearSolverStatus::Breakdown;
            break;
        }
        if (residual_norm <= target) {
            status = LinearSolverStatus::Converged;
            break;
        }
        previous_rho = rho;
    }

    vectorToMatrix(solution, field, mesh);
    exchangeColumns(field, rank, num_procs);
    return finalResult(
        status, iterations, initial_residual, tolerance,
        right_hand_side, solution, distributed_operator);
}
