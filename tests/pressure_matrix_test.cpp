#include "io/mesh_reader.h"
#include "mesh/boundary.h"
#include "numerics/momentum.h"
#include "numerics/pressure_correction.h"
#include "numerics/rhie_chow.h"
#include "parallel/domain_decomposition.h"
#include "solvers/linear_solver.h"

#include <eigen3/Eigen/IterativeLinearSolvers>
#include <mpi.h>

#include <cmath>
#include <iostream>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    if (argc != 2) {
        std::cerr << "usage: pressure_matrix_test <mesh-folder>\n";
        MPI_Finalize();
        return 2;
    }

    Mesh original = readMesh(argv[1]);
    Mesh mesh = extractLocalMesh(original, 0, 1);
    initializeFlowFields(mesh);
    Equation momentum(mesh);
    Equation pressure(mesh);
    Eigen::VectorXd source_v(mesh.internumber);
    assembleMomentum(
        mesh, momentum, source_v, 0.01, 0.5, TimeTerm::none());
    interpolateFaceVelocity(mesh, momentum);
    assemblePressureCorrection(mesh, pressure, momentum, 0, 1);

    Eigen::SparseMatrix<double> transpose = pressure.A.transpose();
    const double symmetry_error = (pressure.A - transpose).norm();
    int outlet_rows = 0;
    int strict_rows = 0;
    double minimum_margin = 1e300;
    for (int n = 0; n < mesh.internumber; ++n) {
        const int i = mesh.interi[static_cast<std::size_t>(n)];
        const int j = mesh.interj[static_cast<std::size_t>(n)];
        const double neighbours =
            (isCoupledCell(mesh.bctype(i, j + 1)) ? pressure.A_e(i, j) : 0.0) +
            (isCoupledCell(mesh.bctype(i, j - 1)) ? pressure.A_w(i, j) : 0.0) +
            (isCoupledCell(mesh.bctype(i - 1, j)) ? pressure.A_n(i, j) : 0.0) +
            (isCoupledCell(mesh.bctype(i + 1, j)) ? pressure.A_s(i, j) : 0.0);
        minimum_margin = std::min(
            minimum_margin, pressure.A_p(i, j) - neighbours);
        if (pressure.A_p(i, j) - neighbours > 1e-14) {
            ++strict_rows;
        }
        if (isPressureOutlet(mesh.bctype(i, j - 1)) ||
            isPressureOutlet(mesh.bctype(i, j + 1)) ||
            isPressureOutlet(mesh.bctype(i - 1, j)) ||
            isPressureOutlet(mesh.bctype(i + 1, j))) {
            ++outlet_rows;
            if (!(pressure.A_p(i, j) > neighbours)) {
                std::cerr << "fixed-pressure coefficient missing from diagonal\n";
                MPI_Finalize();
                return 1;
            }
        }
    }

    Eigen::ConjugateGradient<
        Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper,
        Eigen::DiagonalPreconditioner<double>> eigen_solver;
    eigen_solver.setTolerance(1e-7);
    eigen_solver.setMaxIterations(1000);
    eigen_solver.compute(pressure.A);
    const Eigen::VectorXd solution = eigen_solver.solve(pressure.source);
    const double eigen_relative =
        (pressure.A * solution - pressure.source).norm() /
        std::max(pressure.source.norm(), 1e-30);

    Eigen::MatrixXd field = Eigen::MatrixXd::Zero(mesh.ny, mesh.nx);
    const LinearSolverResult result = solveFieldPCG(
        pressure, pressure.source, mesh, field,
        1e-7, 1000, 0, 1, false);

    std::cout << "symmetry=" << symmetry_error
              << " min_diagonal_margin=" << minimum_margin
              << " outlet_rows=" << outlet_rows
              << " strict_rows=" << strict_rows
              << " eigen_iterations=" << eigen_solver.iterations()
              << " eigen_relative=" << eigen_relative
              << " distributed_iterations=" << result.iterations
              << " distributed_relative=" << result.relative_residual
              << '\n';

    const bool passed = symmetry_error < 1e-12 &&
        minimum_margin >= -1e-12 &&
        result.converged() &&
        ((outlet_rows > 0 && strict_rows == outlet_rows) ||
         (outlet_rows == 0 && strict_rows == 1)) &&
        result.final_residual <= 1e-7 *
            std::max(result.initial_residual, pressure.source.norm());
    MPI_Finalize();
    return passed ? 0 : 1;
}
