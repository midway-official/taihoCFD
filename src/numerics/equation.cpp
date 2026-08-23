#include "numerics/equation.h"

#include <array>
#include <tuple>
#include <vector>

Equation::Equation(Mesh& mesh_value)
    : A_p(Eigen::MatrixXd::Zero(mesh_value.ny, mesh_value.nx)),
      A_e(Eigen::MatrixXd::Zero(mesh_value.ny, mesh_value.nx)),
      A_w(Eigen::MatrixXd::Zero(mesh_value.ny, mesh_value.nx)),
      A_n(Eigen::MatrixXd::Zero(mesh_value.ny, mesh_value.nx)),
      A_s(Eigen::MatrixXd::Zero(mesh_value.ny, mesh_value.nx)),
      source(Eigen::VectorXd::Zero(mesh_value.internumber)),
      A(mesh_value.internumber, mesh_value.internumber),
      mesh(mesh_value)
{}

void Equation::reset() {
    A_p.setZero();
    A_e.setZero();
    A_w.setZero();
    A_n.setZero();
    A_s.setZero();
    source.setZero();
}

void Equation::buildMatrix() {
    const auto visitCoefficients = [&](auto&& add) {
        for (int n = 0; n < mesh.internumber; ++n) {
            const int i = mesh.interi[static_cast<std::size_t>(n)];
            const int j = mesh.interj[static_cast<std::size_t>(n)];
            add(n, n, A_p(i, j));

            const std::array<std::tuple<int, int, double>, 4> neighbours{{
                {i, j + 1, A_e(i, j)},
                {i, j - 1, A_w(i, j)},
                {i - 1, j, A_n(i, j)},
                {i + 1, j, A_s(i, j)},
            }};
            for (const auto& [ni, nj, coefficient] : neighbours) {
                if (isInteriorCell(mesh, ni, nj)) {
                    add(n, mesh.interid(ni, nj), -coefficient);
                }
            }
        }
    };

    if (A.nonZeros() != 0) {
        visitCoefficients([&](int row, int column, double value) {
            A.coeffRef(row, column) = value;
        });
        return;
    }

    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(static_cast<std::size_t>(mesh.internumber) * 5U);
    visitCoefficients([&](int row, int column, double value) {
        entries.emplace_back(row, column, value);
    });
    A.setFromTriplets(entries.begin(), entries.end());
    A.makeCompressed();
}
