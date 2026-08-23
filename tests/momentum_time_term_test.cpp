#include "io/mesh_reader.h"
#include "mesh/boundary.h"
#include "numerics/momentum.h"
#include "numerics/operators.h"
#include "solvers/solution_config.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

double scaledError(double actual, double expected) {
    return std::abs(actual - expected) /
        std::max({1.0, std::abs(actual), std::abs(expected)});
}

}  // namespace

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    try {
        const std::string mesh_folder = argc == 2 ? argv[1] : "poiseuille";
        Mesh mesh = readMesh(mesh_folder);
        initializeFlowFields(mesh);

        constexpr double viscosity = 0.01;
        constexpr double relaxation = 0.6;
        constexpr double dt = 0.025;
        for (int n = 0; n < mesh.internumber; ++n) {
            const int i = mesh.interi[static_cast<std::size_t>(n)];
            const int j = mesh.interj[static_cast<std::size_t>(n)];
            mesh.u0(i, j) = 0.1 + 1e-3 * static_cast<double>(n);
            mesh.v0(i, j) = -0.05 + 5e-4 * static_cast<double>(n);
            mesh.u_star(i, j) = 0.02;
            mesh.v_star(i, j) = -0.01;
        }

        Equation steady(mesh);
        Equation transient(mesh);
        Equation decomposed(mesh);
        Eigen::VectorXd steady_v;
        Eigen::VectorXd transient_v;
        Eigen::VectorXd decomposed_v = Eigen::VectorXd::Zero(mesh.internumber);
        assembleMomentum(
            mesh, steady, steady_v, viscosity, relaxation, TimeTerm::none(),
            NumericalSchemes::steady());
        assembleMomentum(
            mesh, transient, transient_v, viscosity, relaxation,
            TimeTerm::backwardEuler(dt, mesh.u0, mesh.v0),
            NumericalSchemes::backwardEuler());
        decomposed.reset();
        const TimeTerm time_term =
            TimeTerm::backwardEuler(dt, mesh.u0, mesh.v0);
        addDdt(
            mesh, decomposed, decomposed_v, time_term,
            TimeScheme::BackwardEuler);
        addConvection(
            mesh, decomposed, decomposed_v, ConvectionScheme::Upwind);
        addLaplacian(
            mesh, decomposed, decomposed_v, viscosity,
            LaplacianScheme::Orthogonal);
        addPressureGradient(
            mesh, decomposed, decomposed_v, GradientScheme::Central);
        applyVelocityEquationRelaxation(
            mesh, decomposed, decomposed_v, relaxation);

        double maximum_spatial_error = 0.0;
        double maximum_diagonal_error = 0.0;
        double maximum_source_error = 0.0;
        double maximum_composition_error = 0.0;
        for (int n = 0; n < mesh.internumber; ++n) {
            const int i = mesh.interi[static_cast<std::size_t>(n)];
            const int j = mesh.interj[static_cast<std::size_t>(n)];
            maximum_spatial_error = std::max({
                maximum_spatial_error,
                scaledError(transient.A_e(i, j), steady.A_e(i, j)),
                scaledError(transient.A_w(i, j), steady.A_w(i, j)),
                scaledError(transient.A_n(i, j), steady.A_n(i, j)),
                scaledError(transient.A_s(i, j), steady.A_s(i, j)),
            });

            const double time_diagonal = mesh.vol(i, j) / dt;
            maximum_diagonal_error = std::max(
                maximum_diagonal_error,
                scaledError(
                    transient.A_p(i, j) - steady.A_p(i, j),
                    time_diagonal));

            const double expected_u_source =
                relaxation * time_diagonal * mesh.u0(i, j) +
                (1.0 - relaxation) * time_diagonal * mesh.u_star(i, j);
            const double expected_v_source =
                relaxation * time_diagonal * mesh.v0(i, j) +
                (1.0 - relaxation) * time_diagonal * mesh.v_star(i, j);
            maximum_source_error = std::max({
                maximum_source_error,
                scaledError(
                    transient.source[n] - steady.source[n],
                    expected_u_source),
                scaledError(
                    transient_v[n] - steady_v[n], expected_v_source),
            });
            maximum_composition_error = std::max({
                maximum_composition_error,
                scaledError(decomposed.A_p(i, j), transient.A_p(i, j)),
                scaledError(decomposed.A_e(i, j), transient.A_e(i, j)),
                scaledError(decomposed.A_w(i, j), transient.A_w(i, j)),
                scaledError(decomposed.A_n(i, j), transient.A_n(i, j)),
                scaledError(decomposed.A_s(i, j), transient.A_s(i, j)),
                scaledError(decomposed.source[n], transient.source[n]),
                scaledError(decomposed_v[n], transient_v[n]),
            });
        }

        constexpr double tolerance = 1e-12;
        std::cout << "spatial_error=" << maximum_spatial_error
                  << " diagonal_error=" << maximum_diagonal_error
                  << " source_error=" << maximum_source_error
                  << " composition_error=" << maximum_composition_error
                  << '\n';
        if (maximum_spatial_error > tolerance ||
            maximum_diagonal_error > tolerance ||
            maximum_source_error > tolerance ||
            maximum_composition_error > tolerance) {
            throw std::runtime_error("定常/非定常动量装配回归失败");
        }

        bool invalid_solver_rejected = false;
        try {
            SolutionConfig invalid;
            invalid.velocity.solver = LinearSolverType::PCG;
            invalid.validate();
        } catch (const std::invalid_argument&) {
            invalid_solver_rejected = true;
        }
        bool invalid_scheme_rejected = false;
        try {
            NumericalSchemes invalid;
            invalid.pressure_gradient = static_cast<GradientScheme>(99);
            invalid.validate();
        } catch (const std::invalid_argument&) {
            invalid_scheme_rejected = true;
        }
        if (!invalid_solver_rejected || !invalid_scheme_rejected) {
            throw std::runtime_error("非法离散或求解配置未被拒绝");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        MPI_Finalize();
        return 1;
    }
    MPI_Finalize();
    return 0;
}
