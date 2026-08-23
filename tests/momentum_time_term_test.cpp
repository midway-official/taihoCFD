#include "io/mesh_reader.h"
#include "mesh/boundary.h"
#include "numerics/momentum.h"

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
        Eigen::VectorXd steady_v;
        Eigen::VectorXd transient_v;
        assembleMomentum(
            mesh, steady, steady_v, viscosity, relaxation, TimeTerm::none());
        assembleMomentum(
            mesh, transient, transient_v, viscosity, relaxation,
            TimeTerm::backwardEuler(dt, mesh.u0, mesh.v0));

        double maximum_spatial_error = 0.0;
        double maximum_diagonal_error = 0.0;
        double maximum_source_error = 0.0;
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
        }

        constexpr double tolerance = 1e-12;
        std::cout << "spatial_error=" << maximum_spatial_error
                  << " diagonal_error=" << maximum_diagonal_error
                  << " source_error=" << maximum_source_error << '\n';
        if (maximum_spatial_error > tolerance ||
            maximum_diagonal_error > tolerance ||
            maximum_source_error > tolerance) {
            throw std::runtime_error("定常/非定常动量装配回归失败");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        MPI_Finalize();
        return 1;
    }
    MPI_Finalize();
    return 0;
}
