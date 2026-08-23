#include "numerics/pressure_correction.h"

#include "numerics/stencil.h"

#include <mpi.h>

#include <algorithm>
#include <stdexcept>

void assemblePressureCorrection(
    Mesh& mesh,
    Equation& pressure,
    const Equation& momentum,
    int rank,
    int num_procs)
{
    pressure.reset();
    mesh.p_prime.setZero();

    int local_has_pressure_outlet = 0;
    for (int j = mesh.owned_j_begin; j < mesh.owned_j_end; ++j) {
        for (int i = 0; i < mesh.ny; ++i) {
            local_has_pressure_outlet = std::max(
                local_has_pressure_outlet,
                isPressureOutlet(mesh.bctype(i, j)) ? 1 : 0);
        }
    }
    int global_has_pressure_outlet = 0;
    MPI_Allreduce(
        &local_has_pressure_outlet,
        &global_has_pressure_outlet,
        1,
        MPI_INT,
        MPI_MAX,
        MPI_COMM_WORLD);

    for (int n = 0; n < mesh.internumber; ++n) {
        const int i = mesh.interi[static_cast<std::size_t>(n)];
        const int j = mesh.interj[static_cast<std::size_t>(n)];
        double diagonal = 0.0;

        const auto addXFace = [&](
            int neighbour_j,
            double area,
            double distance,
            double& coefficient)
        {
            const int type = mesh.bctype(i, neighbour_j);
            if (isCoupledCell(type)) {
                coefficient = area * stencil::xFaceMobility(
                    mesh, momentum, i, std::min(j, neighbour_j));
                diagonal += coefficient;
            } else if (isPressureOutlet(type)) {
                // fixedValue p'=0: the boundary term contributes only to aP.
                coefficient = area * stencil::pressureBoundaryMobility(
                    mesh, momentum, i, j, distance);
                diagonal += coefficient;
            } else {
                // Wall and velocity inlet use zeroGradient p'.
                coefficient = 0.0;
            }
        };

        const auto addYFace = [&](
            int neighbour_i,
            double area,
            double distance,
            double& coefficient)
        {
            const int type = mesh.bctype(neighbour_i, j);
            if (isCoupledCell(type)) {
                coefficient = area * stencil::yFaceMobility(
                    mesh, momentum, std::min(i, neighbour_i), j);
                diagonal += coefficient;
            } else if (isPressureOutlet(type)) {
                coefficient = area * stencil::pressureBoundaryMobility(
                    mesh, momentum, i, j, distance);
                diagonal += coefficient;
            } else {
                coefficient = 0.0;
            }
        };

        const double de = stencil::checkedDistance(
            mesh.x_c(i, j + 1) - mesh.x_c(i, j), "p de", i, j);
        const double dw = stencil::checkedDistance(
            mesh.x_c(i, j) - mesh.x_c(i, j - 1), "p dw", i, j);
        const double dn = stencil::checkedDistance(
            mesh.y_c(i, j) - mesh.y_c(i - 1, j), "p dn", i, j);
        const double ds = stencil::checkedDistance(
            mesh.y_c(i + 1, j) - mesh.y_c(i, j), "p ds", i, j);

        addXFace(j + 1, mesh.area_e(i, j), de, pressure.A_e(i, j));
        addXFace(j - 1, mesh.area_w(i, j), dw, pressure.A_w(i, j));
        addYFace(i - 1, mesh.area_n(i, j), dn, pressure.A_n(i, j));
        addYFace(i + 1, mesh.area_s(i, j), ds, pressure.A_s(i, j));

        pressure.A_p(i, j) = diagonal;
        pressure.source[n] =
            -(mesh.u_face(i, j) * mesh.area_e(i, j) -
              mesh.u_face(i, j - 1) * mesh.area_w(i, j))
            -(mesh.v_face(i, j) * mesh.area_s(i, j) -
              mesh.v_face(i - 1, j) * mesh.area_n(i, j));
    }

    if (!global_has_pressure_outlet && rank == 0) {
        if (mesh.internumber == 0) {
            throw std::runtime_error("rank 0 没有可用于压力参考的内部单元");
        }
        // OpenFOAM fvMatrix::setReference semantics for reference value zero.
        const int i = mesh.interi.front();
        const int j = mesh.interj.front();
        const int n = mesh.interid(i, j);
        const double old_diagonal = pressure.A_p(i, j);
        constexpr double reference_value = 0.0;
        pressure.source[n] += old_diagonal * reference_value;
        pressure.A_p(i, j) += old_diagonal;
    }

    pressure.buildMatrix();
    (void)num_procs;
}

void correctPressure(Mesh& mesh, double pressure_relaxation) {
    if (!(pressure_relaxation > 0.0 && pressure_relaxation <= 1.0)) {
        throw std::invalid_argument("压力松弛因子必须在 (0, 1] 内");
    }

    mesh.p_star = mesh.p;
    for (int n = 0; n < mesh.internumber; ++n) {
        const int i = mesh.interi[static_cast<std::size_t>(n)];
        const int j = mesh.interj[static_cast<std::size_t>(n)];
        mesh.p_star(i, j) =
            mesh.p(i, j) + pressure_relaxation * mesh.p_prime(i, j);
    }
    mesh.p = mesh.p_star;

    for (int j = 0; j < mesh.nx; ++j) {
        for (int i = 0; i < mesh.ny; ++i) {
            if (isPressureOutlet(mesh.bctype(i, j))) {
                mesh.p(i, j) = 0.0;
                mesh.p_star(i, j) = 0.0;
                mesh.p_prime(i, j) = 0.0;
            }
        }
    }
}

void correctVelocity(Mesh& mesh, const Equation& momentum) {
    for (int n = 0; n < mesh.internumber; ++n) {
        const int i = mesh.interi[static_cast<std::size_t>(n)];
        const int j = mesh.interj[static_cast<std::size_t>(n)];
        const double de = stencil::checkedDistance(
            mesh.x_c(i, j + 1) - mesh.x_c(i, j), "u de", i, j);
        const double dw = stencil::checkedDistance(
            mesh.x_c(i, j) - mesh.x_c(i, j - 1), "u dw", i, j);
        const double dn = stencil::checkedDistance(
            mesh.y_c(i, j) - mesh.y_c(i - 1, j), "v dn", i, j);
        const double ds = stencil::checkedDistance(
            mesh.y_c(i + 1, j) - mesh.y_c(i, j), "v ds", i, j);

        const double p_w = stencil::pressureCorrectionSample(
            mesh, i, j - 1, i, j);
        const double p_e = stencil::pressureCorrectionSample(
            mesh, i, j + 1, i, j);
        const double p_n = stencil::pressureCorrectionSample(
            mesh, i - 1, j, i, j);
        const double p_s = stencil::pressureCorrectionSample(
            mesh, i + 1, j, i, j);

        const double mobility = mesh.vol(i, j) / momentum.A_p(i, j);
        mesh.u_star(i, j) = mesh.u(i, j) + mobility * (p_w - p_e) / (dw + de);
        mesh.v_star(i, j) = mesh.v(i, j) + mobility * (p_n - p_s) / (dn + ds);
    }

    for (int j = 0; j < mesh.nx - 1; ++j) {
        for (int i = 0; i < mesh.ny; ++i) {
            const int left = mesh.bctype(i, j);
            const int right = mesh.bctype(i, j + 1);
            if (isCoupledCell(left) && isCoupledCell(right) &&
                (isInterior(left) || isInterior(right))) {
                mesh.u_face(i, j) +=
                    stencil::xFaceMobility(mesh, momentum, i, j) *
                    (mesh.p_prime(i, j) - mesh.p_prime(i, j + 1));
            } else if (isPressureOutlet(right) && isInterior(left)) {
                const double distance = mesh.x_c(i, j + 1) - mesh.x_c(i, j);
                mesh.u_face(i, j) += stencil::pressureBoundaryMobility(
                    mesh, momentum, i, j, distance) * mesh.p_prime(i, j);
            } else if (isPressureOutlet(left) && isInterior(right)) {
                const double distance = mesh.x_c(i, j + 1) - mesh.x_c(i, j);
                mesh.u_face(i, j) -= stencil::pressureBoundaryMobility(
                    mesh, momentum, i, j + 1, distance) *
                    mesh.p_prime(i, j + 1);
            }
        }
    }

    for (int j = 0; j < mesh.nx; ++j) {
        for (int i = 0; i < mesh.ny - 1; ++i) {
            const int lower = mesh.bctype(i, j);
            const int upper = mesh.bctype(i + 1, j);
            if (isCoupledCell(lower) && isCoupledCell(upper) &&
                (isInterior(lower) || isInterior(upper))) {
                mesh.v_face(i, j) +=
                    stencil::yFaceMobility(mesh, momentum, i, j) *
                    (mesh.p_prime(i, j) - mesh.p_prime(i + 1, j));
            } else if (isPressureOutlet(upper) && isInterior(lower)) {
                const double distance = mesh.y_c(i + 1, j) - mesh.y_c(i, j);
                mesh.v_face(i, j) += stencil::pressureBoundaryMobility(
                    mesh, momentum, i, j, distance) * mesh.p_prime(i, j);
            } else if (isPressureOutlet(lower) && isInterior(upper)) {
                const double distance = mesh.y_c(i + 1, j) - mesh.y_c(i, j);
                mesh.v_face(i, j) -= stencil::pressureBoundaryMobility(
                    mesh, momentum, i + 1, j, distance) *
                    mesh.p_prime(i + 1, j);
            }
        }
    }
}
