#include "numerics/momentum.h"

#include "mesh/boundary.h"
#include "numerics/stencil.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

struct FaceContribution {
    double neighbour = 0.0;
    double diagonal = 0.0;
    double source_u = 0.0;
    double source_v = 0.0;
};

FaceContribution assembleFaceContribution(
    const Mesh& mesh,
    int owner_i,
    int owner_j,
    int neighbour_i,
    int neighbour_j,
    double diffusion,
    double outward_flux)
{
    FaceContribution result;
    const int type = mesh.bctype(neighbour_i, neighbour_j);
    const double outflow = std::max(outward_flux, 0.0);
    const double inflow = std::max(-outward_flux, 0.0);

    if (isCoupledCell(type)) {
        result.neighbour = diffusion + inflow;
        result.diagonal = diffusion + outflow;
    } else if (isWall(type) || isVelocityInlet(type)) {
        const double diffusion_scale = isWall(type) ? 2.0 : 1.0;
        const double source_coefficient = diffusion_scale * diffusion + inflow;
        result.diagonal = diffusion_scale * diffusion + outflow;
        result.source_u =
            source_coefficient * boundaryU(mesh, neighbour_i, neighbour_j);
        result.source_v =
            source_coefficient * boundaryV(mesh, neighbour_i, neighbour_j);
    } else if (isPressureOutlet(type)) {
        result.diagonal = diffusion + outflow;
        result.source_u =
            (diffusion + inflow) * mesh.u_star(owner_i, owner_j);
        result.source_v =
            (diffusion + inflow) * mesh.v_star(owner_i, owner_j);
    } else {
        throw std::runtime_error("未知边界类型: " + std::to_string(type));
    }
    return result;
}

}  // namespace

TimeTerm TimeTerm::none() {
    return {};
}

TimeTerm TimeTerm::backwardEuler(
    double dt_value,
    const Eigen::MatrixXd& old_u,
    const Eigen::MatrixXd& old_v)
{
    if (!(dt_value > 0.0) || !std::isfinite(dt_value)) {
        throw std::invalid_argument("时间步长必须为正有限数");
    }
    return {dt_value, &old_u, &old_v};
}

void assembleMomentum(
    Mesh& mesh,
    Equation& momentum,
    Eigen::VectorXd& source_v,
    double viscosity,
    double velocity_relaxation,
    const TimeTerm& time_term)
{
    if (!(viscosity > 0.0) || !std::isfinite(viscosity)) {
        throw std::invalid_argument("动力粘度必须为正且有限");
    }
    if (!(velocity_relaxation > 0.0 && velocity_relaxation <= 1.0)) {
        throw std::invalid_argument("速度松弛因子必须在 (0, 1] 内");
    }
    if (!std::isfinite(time_term.dt) || time_term.dt < 0.0) {
        throw std::invalid_argument("时间项 dt 必须为非负有限数");
    }
    if (time_term.enabled() &&
        (time_term.u_previous == nullptr || time_term.v_previous == nullptr ||
         time_term.u_previous->rows() != mesh.ny ||
         time_term.u_previous->cols() != mesh.nx ||
         time_term.v_previous->rows() != mesh.ny ||
         time_term.v_previous->cols() != mesh.nx)) {
        throw std::invalid_argument("非定常历史速度场尺寸不匹配");
    }

    momentum.reset();
    source_v.setZero(mesh.internumber);

    for (int n = 0; n < mesh.internumber; ++n) {
        const int i = mesh.interi[static_cast<std::size_t>(n)];
        const int j = mesh.interj[static_cast<std::size_t>(n)];

        const double de = stencil::checkedDistance(
            mesh.x_c(i, j + 1) - mesh.x_c(i, j), "de", i, j);
        const double dw = stencil::checkedDistance(
            mesh.x_c(i, j) - mesh.x_c(i, j - 1), "dw", i, j);
        const double ds = stencil::checkedDistance(
            mesh.y_c(i + 1, j) - mesh.y_c(i, j), "ds", i, j);
        const double dn = stencil::checkedDistance(
            mesh.y_c(i, j) - mesh.y_c(i - 1, j), "dn", i, j);

        const std::array<double, 4> diffusion{{
            mesh.area_e(i, j) * viscosity / de,
            mesh.area_w(i, j) * viscosity / dw,
            mesh.area_n(i, j) * viscosity / dn,
            mesh.area_s(i, j) * viscosity / ds,
        }};
        const std::array<double, 4> outward_flux{{
            mesh.area_e(i, j) * mesh.u_face(i, j),
            -mesh.area_w(i, j) * mesh.u_face(i, j - 1),
            -mesh.area_n(i, j) * mesh.v_face(i - 1, j),
            mesh.area_s(i, j) * mesh.v_face(i, j),
        }};
        const std::array<std::pair<int, int>, 4> neighbours{{
            {i, j + 1}, {i, j - 1}, {i - 1, j}, {i + 1, j},
        }};

        std::array<FaceContribution, 4> faces;
        for (std::size_t face = 0; face < faces.size(); ++face) {
            faces[face] = assembleFaceContribution(
                mesh, i, j,
                neighbours[face].first, neighbours[face].second,
                diffusion[face], outward_flux[face]);
        }

        momentum.A_e(i, j) = velocity_relaxation * faces[0].neighbour;
        momentum.A_w(i, j) = velocity_relaxation * faces[1].neighbour;
        momentum.A_n(i, j) = velocity_relaxation * faces[2].neighbour;
        momentum.A_s(i, j) = velocity_relaxation * faces[3].neighbour;

        double diagonal = 0.0;
        double source_u =
            (stencil::pressureSample(mesh, i, j - 1, i, j) -
             stencil::pressureSample(mesh, i, j + 1, i, j)) *
            mesh.vol(i, j) / (dw + de);
        double source_for_v =
            (stencil::pressureSample(mesh, i - 1, j, i, j) -
             stencil::pressureSample(mesh, i + 1, j, i, j)) *
            mesh.vol(i, j) / (dn + ds);
        for (const FaceContribution& face : faces) {
            diagonal += face.diagonal;
            source_u += face.source_u;
            source_for_v += face.source_v;
        }

        if (time_term.enabled()) {
            diagonal += mesh.vol(i, j) / time_term.dt;
            source_u += mesh.vol(i, j) * (*time_term.u_previous)(i, j) /
                time_term.dt;
            source_for_v += mesh.vol(i, j) * (*time_term.v_previous)(i, j) /
                time_term.dt;
        }

        momentum.A_p(i, j) = diagonal;
        momentum.source[n] =
            velocity_relaxation * source_u +
            (1.0 - velocity_relaxation) * diagonal * mesh.u_star(i, j);
        source_v[n] =
            velocity_relaxation * source_for_v +
            (1.0 - velocity_relaxation) * diagonal * mesh.v_star(i, j);

        stencil::requireFinite(momentum.A_p(i, j), "动量对角系数", i, j);
        stencil::requireFinite(momentum.source[n], "u 动量源项", i, j);
        stencil::requireFinite(source_v[n], "v 动量源项", i, j);
    }

    momentum.buildMatrix();
}
