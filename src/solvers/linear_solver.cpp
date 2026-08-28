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

double globalDot(
    const Eigen::VectorXd& left,
    const Eigen::VectorXd& right,
    MPI_Comm communicator)
{
    const double local = left.dot(right);
    double global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, communicator);
    return global;
}

double globalNorm(const Eigen::VectorXd& value, MPI_Comm communicator) {
    return std::sqrt(std::max(globalDot(value, value, communicator), 0.0));
}

class DistributedOperator {
public:
    DistributedOperator(
        const Equation& equation,
        const Mesh& mesh,
        const ParallelContext& parallel)
        : equation_(equation),
          mesh_(mesh),
          parallel_(parallel),
          exchange_field_(Eigen::MatrixXd::Zero(mesh.ny, mesh.nx))
    {}

    void apply(const Eigen::VectorXd& input, Eigen::VectorXd& output) {
        output.noalias() = equation_.A * input;
        vectorToMatrix(input, exchange_field_, mesh_);
        exchangeColumns(exchange_field_, parallel_);

        for (int n = 0; n < mesh_.internumber; ++n) {
            const int i = mesh_.interi[static_cast<std::size_t>(n)];
            const int j = mesh_.interj[static_cast<std::size_t>(n)];
            if (isProcessorCell(mesh_, i, j + 1)) {
                output[n] -= equation_.A_e(i, j) * exchange_field_(i, j + 1);
            }
            if (isProcessorCell(mesh_, i, j - 1)) {
                output[n] -= equation_.A_w(i, j) * exchange_field_(i, j - 1);
            }
        }
    }

private:
    const Equation& equation_;
    const Mesh& mesh_;
    const ParallelContext& parallel_;
    Eigen::MatrixXd exchange_field_;
};

double convergenceScale(double initial_residual, double right_hand_side_norm) {
    return std::max({initial_residual, right_hand_side_norm, 1e-30});
}

LinearSolverResult finalResult(
    LinearSolverStatus status,
    int iterations,
    double initial_residual,
    const LinearSolverConfig& config,
    const Eigen::VectorXd& right_hand_side,
    const Eigen::VectorXd& solution,
    DistributedOperator& distributed_operator,
    const ParallelContext& parallel)
{
    Eigen::VectorXd matrix_solution(solution.size());
    distributed_operator.apply(solution, matrix_solution);
    const double final_residual = globalNorm(
        right_hand_side - matrix_solution, parallel.communicator);
    const double right_hand_side_norm = globalNorm(
        right_hand_side, parallel.communicator);
    const double reporting_scale = std::max(
        convergenceScale(initial_residual, right_hand_side_norm),
        config.absolute_tolerance / config.relative_tolerance);

    return {
        status,
        iterations,
        initial_residual,
        final_residual,
        final_residual / reporting_scale,
    };
}

class LinearSolveWorkspace {
public:
    LinearSolveWorkspace(
        const Equation& equation,
        const Eigen::VectorXd& right_hand_side_value,
        const Mesh& mesh_value,
        Eigen::MatrixXd& field_value,
        const LinearSolverConfig& config_value,
        const ParallelContext& parallel_value,
        bool warm_start)
        : right_hand_side(right_hand_side_value),
          mesh(mesh_value),
          field(field_value),
          config(config_value),
          parallel(parallel_value),
          distributed_operator(equation, mesh, parallel),
          solution(mesh.internumber),
          residual(mesh.internumber)
    {
        parallel.validate();
        if (warm_start) {
            matrixToVector(field, solution, mesh);
        } else {
            solution.setZero();
        }

        Eigen::VectorXd matrix_solution(mesh.internumber);
        distributed_operator.apply(solution, matrix_solution);
        residual = right_hand_side - matrix_solution;
        initial_residual = globalNorm(residual, parallel.communicator);
        const double scale = convergenceScale(
            initial_residual,
            globalNorm(right_hand_side, parallel.communicator));
        target = std::max(
            config.relative_tolerance * scale, config.absolute_tolerance);
    }

    LinearSolverResult finish(LinearSolverStatus status, int iterations) {
        vectorToMatrix(solution, field, mesh);
        exchangeColumns(field, parallel);
        return finalResult(
            status, iterations, initial_residual, config,
            right_hand_side, solution, distributed_operator, parallel);
    }

    const Eigen::VectorXd& right_hand_side;
    const Mesh& mesh;
    Eigen::MatrixXd& field;
    const LinearSolverConfig& config;
    const ParallelContext& parallel;
    DistributedOperator distributed_operator;
    Eigen::VectorXd solution;
    Eigen::VectorXd residual;
    double initial_residual = 0.0;
    double target = 0.0;
};

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
    const LinearSolverConfig& config,
    const ParallelContext& parallel,
    bool warm_start)
{
    config.validate();
    if (config.solver != LinearSolverType::PCG ||
        right_hand_side.size() != mesh.internumber) {
        throw std::invalid_argument("PCG 参数无效");
    }

    LinearSolveWorkspace workspace(
        equation, right_hand_side, mesh, field, config, parallel, warm_start);

    Eigen::IncompleteCholesky<double> preconditioner;
    preconditioner.compute(equation.A);
    if (preconditioner.info() != Eigen::Success) {
        throw std::runtime_error("压力矩阵的 incomplete-Cholesky 分解失败");
    }
    Eigen::VectorXd preconditioned =
        preconditioner.solve(workspace.residual);
    Eigen::VectorXd direction = preconditioned;
    Eigen::VectorXd matrix_direction(mesh.internumber);

    if (workspace.initial_residual <= workspace.target) {
        return workspace.finish(LinearSolverStatus::Converged, 0);
    }

    double residual_preconditioned =
        globalDot(
            workspace.residual, preconditioned, parallel.communicator);
    LinearSolverStatus status = LinearSolverStatus::MaxIterations;
    int iterations = 0;

    for (int iteration = 1; iteration <= config.max_iterations; ++iteration) {
        workspace.distributed_operator.apply(direction, matrix_direction);
        const double direction_matrix_direction =
            globalDot(direction, matrix_direction, parallel.communicator);
        if (invalidScalar(direction_matrix_direction) ||
            direction_matrix_direction <= breakdown_tolerance ||
            invalidScalar(residual_preconditioned)) {
            status = LinearSolverStatus::Breakdown;
            iterations = iteration - 1;
            break;
        }

        const double alpha = residual_preconditioned /
            direction_matrix_direction;
        workspace.solution.noalias() += alpha * direction;
        workspace.residual.noalias() -= alpha * matrix_direction;
        const double residual_norm = globalNorm(
            workspace.residual, parallel.communicator);
        iterations = iteration;
        if (invalidScalar(residual_norm)) {
            status = LinearSolverStatus::Breakdown;
            break;
        }
        if (residual_norm <= workspace.target) {
            status = LinearSolverStatus::Converged;
            break;
        }

        preconditioned = preconditioner.solve(workspace.residual);
        if (preconditioner.info() != Eigen::Success) {
            status = LinearSolverStatus::Breakdown;
            break;
        }
        const double next_residual_preconditioned =
            globalDot(
                workspace.residual, preconditioned, parallel.communicator);
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

    return workspace.finish(status, iterations);
}

LinearSolverResult solveFieldBiCGSTAB(
    const Equation& equation,
    const Eigen::VectorXd& right_hand_side,
    const Mesh& mesh,
    Eigen::MatrixXd& field,
    const LinearSolverConfig& config,
    const ParallelContext& parallel,
    bool warm_start)
{
    config.validate();
    if (config.solver != LinearSolverType::BiCGSTAB ||
        right_hand_side.size() != mesh.internumber) {
        throw std::invalid_argument("BiCGSTAB 参数无效");
    }

    LinearSolveWorkspace workspace(
        equation, right_hand_side, mesh, field, config, parallel, warm_start);
    const Eigen::VectorXd shadow_residual = workspace.residual;

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

    if (workspace.initial_residual <= workspace.target) {
        return workspace.finish(LinearSolverStatus::Converged, 0);
    }

    double previous_rho = 1.0;
    double alpha = 1.0;
    double omega = 1.0;
    LinearSolverStatus status = LinearSolverStatus::MaxIterations;
    int iterations = 0;

    for (int iteration = 1; iteration <= config.max_iterations; ++iteration) {
        const double rho = globalDot(
            shadow_residual, workspace.residual, parallel.communicator);
        if (invalidScalar(rho) || std::abs(rho) <= breakdown_tolerance ||
            invalidScalar(omega) || std::abs(omega) <= breakdown_tolerance) {
            status = LinearSolverStatus::Breakdown;
            iterations = iteration - 1;
            break;
        }

        const double beta = (rho / previous_rho) * (alpha / omega);
        direction = workspace.residual +
            beta * (direction - omega * matrix_direction);
        preconditioned_direction = preconditioner.solve(direction);
        if (preconditioner.info() != Eigen::Success) {
            status = LinearSolverStatus::Breakdown;
            iterations = iteration - 1;
            break;
        }
        workspace.distributed_operator.apply(
            preconditioned_direction, matrix_direction);

        const double shadow_matrix_direction =
            globalDot(
                shadow_residual, matrix_direction, parallel.communicator);
        if (invalidScalar(shadow_matrix_direction) ||
            std::abs(shadow_matrix_direction) <= breakdown_tolerance) {
            status = LinearSolverStatus::Breakdown;
            iterations = iteration - 1;
            break;
        }
        alpha = rho / shadow_matrix_direction;
        intermediate = workspace.residual - alpha * matrix_direction;

        const double intermediate_norm = globalNorm(
            intermediate, parallel.communicator);
        if (invalidScalar(intermediate_norm)) {
            status = LinearSolverStatus::Breakdown;
            iterations = iteration;
            break;
        }
        if (intermediate_norm <= workspace.target) {
            workspace.solution.noalias() += alpha * preconditioned_direction;
            workspace.residual = intermediate;
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
        workspace.distributed_operator.apply(
            preconditioned_intermediate, matrix_intermediate);

        const double local_products[2] = {
            matrix_intermediate.dot(intermediate),
            matrix_intermediate.squaredNorm(),
        };
        double global_products[2] = {0.0, 0.0};
        MPI_Allreduce(
            local_products, global_products, 2, MPI_DOUBLE, MPI_SUM,
            parallel.communicator);
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

        workspace.solution.noalias() +=
            alpha * preconditioned_direction +
            omega * preconditioned_intermediate;
        workspace.residual = intermediate - omega * matrix_intermediate;
        const double residual_norm = globalNorm(
            workspace.residual, parallel.communicator);
        iterations = iteration;
        if (invalidScalar(residual_norm)) {
            status = LinearSolverStatus::Breakdown;
            break;
        }
        if (residual_norm <= workspace.target) {
            status = LinearSolverStatus::Converged;
            break;
        }
        previous_rho = rho;
    }

    return workspace.finish(status, iterations);
}

LinearSolverResult solveField(
    const Equation& equation,
    const Eigen::VectorXd& right_hand_side,
    const Mesh& mesh,
    Eigen::MatrixXd& field,
    const LinearSolverConfig& config,
    const ParallelContext& parallel)
{
    config.validate();
    switch (config.solver) {
        case LinearSolverType::Unset:
            break;
        case LinearSolverType::BiCGSTAB:
            return solveFieldBiCGSTAB(
                equation, right_hand_side, mesh, field, config,
                parallel, config.warm_start.value());
        case LinearSolverType::PCG:
            return solveFieldPCG(
                equation, right_hand_side, mesh, field, config,
                parallel, config.warm_start.value());
    }
    throw std::invalid_argument("未知线性求解器类型");
}

LinearSolverResult solveField(
    const Equation& equation,
    const Eigen::VectorXd& right_hand_side,
    const Mesh& mesh,
    Eigen::MatrixXd& field,
    const LinearSolverConfig& config,
    int rank,
    int num_procs)
{
    return solveField(
        equation, right_hand_side, mesh, field, config,
        ParallelContext{MPI_COMM_WORLD, rank, num_procs});
}
