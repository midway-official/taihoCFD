#include "numerics/operators.h"

#include "mesh/boundary.h"
#include "numerics/stencil.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace {

enum class Direction : std::size_t {
    East,
    West,
    North,
    South,
};

struct FaceData {
    int neighbour_i = 0;
    int neighbour_j = 0;
    double area = 0.0;
    double distance = 0.0;
    double outward_flux = 0.0;
};

std::array<FaceData, 4> cellFaces(const Mesh& mesh, int i, int j) {
    const double de = stencil::checkedDistance(
        mesh.x_c(i, j + 1) - mesh.x_c(i, j), "de", i, j);
    const double dw = stencil::checkedDistance(
        mesh.x_c(i, j) - mesh.x_c(i, j - 1), "dw", i, j);
    const double dn = stencil::checkedDistance(
        mesh.y_c(i, j) - mesh.y_c(i - 1, j), "dn", i, j);
    const double ds = stencil::checkedDistance(
        mesh.y_c(i + 1, j) - mesh.y_c(i, j), "ds", i, j);
    return {{
        {i, j + 1, mesh.area_e(i, j), de,
         mesh.area_e(i, j) * mesh.u_face(i, j)},
        {i, j - 1, mesh.area_w(i, j), dw,
         -mesh.area_w(i, j) * mesh.u_face(i, j - 1)},
        {i - 1, j, mesh.area_n(i, j), dn,
         -mesh.area_n(i, j) * mesh.v_face(i - 1, j)},
        {i + 1, j, mesh.area_s(i, j), ds,
         mesh.area_s(i, j) * mesh.v_face(i, j)},
    }};
}

double& neighbourCoefficient(
    Equation& equation,
    Direction direction,
    int i,
    int j)
{
    switch (direction) {
        case Direction::East:
            return equation.A_e(i, j);
        case Direction::West:
            return equation.A_w(i, j);
        case Direction::North:
            return equation.A_n(i, j);
        case Direction::South:
            return equation.A_s(i, j);
    }
    throw std::runtime_error("未知面方向");
}

void addBoundaryConvection(
    const Mesh& mesh,
    const FaceData& face,
    int owner_i,
    int owner_j,
    double inflow,
    double& source_u,
    double& source_v)
{
    const BoundaryPatch& patch = boundaryPatch(
        mesh, face.neighbour_i, face.neighbour_j);
    const Eigen::Vector2d owner{
        mesh.u_star(owner_i, owner_j),
        mesh.v_star(owner_i, owner_j),
    };
    const Eigen::Vector2d boundary = evaluateVelocityBoundary(
        patch, owner, face.outward_flux);
    source_u += inflow * boundary.x();
    source_v += inflow * boundary.y();
}

void addBoundaryDiffusion(
    const Mesh& mesh,
    const FaceData& face,
    int owner_i,
    int owner_j,
    double diffusion,
    double& diagonal,
    double& source_u,
    double& source_v)
{
    const BoundaryPatch& patch = boundaryPatch(
        mesh, face.neighbour_i, face.neighbour_j);
    const bool inlet_part =
        patch.velocity.type == BoundaryConditionType::InletOutlet &&
        face.outward_flux < 0.0;
    const bool fixed = fixesValue(patch.velocity.type) || inlet_part;
    const double scale = fixed && patch.kind == PatchKind::Wall ? 2.0 : 1.0;
    const double coefficient = scale * diffusion;
    const Eigen::Vector2d owner{
        mesh.u_star(owner_i, owner_j),
        mesh.v_star(owner_i, owner_j),
    };
    const Eigen::Vector2d boundary = evaluateVelocityBoundary(
        patch, owner, face.outward_flux);
    diagonal += coefficient;
    source_u += coefficient * boundary.x();
    source_v += coefficient * boundary.y();
}

void requireImplemented(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

void requirePositiveFinite(double value, const char* name) {
    if (!(value > 0.0) || !std::isfinite(value)) {
        throw std::invalid_argument(std::string(name) + "必须为正有限数");
    }
}

}  // namespace

void addDdt(
    Mesh& mesh,
    Equation& equation,
    Eigen::VectorXd& source_v,
    const TimeTerm& time_term,
    double rho,
    TimeScheme scheme)
{
    if (scheme == TimeScheme::SteadyState) {
        if (time_term.enabled()) {
            throw std::invalid_argument("steadyState 格式不能传入时间项");
        }
        return;
    }
    requireImplemented(
        scheme == TimeScheme::BackwardEuler,
        "尚未实现指定的时间格式");
    if (!time_term.enabled() || time_term.u_previous == nullptr ||
        time_term.v_previous == nullptr ||
        time_term.u_previous->rows() != mesh.ny ||
        time_term.u_previous->cols() != mesh.nx ||
        time_term.v_previous->rows() != mesh.ny ||
        time_term.v_previous->cols() != mesh.nx) {
        throw std::invalid_argument("backwardEuler 历史速度场或 dt 无效");
    }
    requirePositiveFinite(rho, "rho ");
    for (int n = 0; n < mesh.internumber; ++n) {
        const int i = mesh.interi[static_cast<std::size_t>(n)];
        const int j = mesh.interj[static_cast<std::size_t>(n)];
        const double coefficient = rho * mesh.vol(i, j) / time_term.dt;
        equation.A_p(i, j) += coefficient;
        equation.source[n] += coefficient * (*time_term.u_previous)(i, j);
        source_v[n] += coefficient * (*time_term.v_previous)(i, j);
    }
}

void addConvection(
    Mesh& mesh,
    Equation& equation,
    Eigen::VectorXd& source_v,
    double rho,
    ConvectionScheme scheme)
{
    requireImplemented(
        scheme == ConvectionScheme::Upwind,
        "尚未实现指定的对流格式");
    requirePositiveFinite(rho, "rho ");
    for (int n = 0; n < mesh.internumber; ++n) {
        const int i = mesh.interi[static_cast<std::size_t>(n)];
        const int j = mesh.interj[static_cast<std::size_t>(n)];
        const auto faces = cellFaces(mesh, i, j);
        for (std::size_t face_index = 0; face_index < faces.size(); ++face_index) {
            const FaceData& face = faces[face_index];
            const double mass_flux = rho * face.outward_flux;
            const double outflow = std::max(mass_flux, 0.0);
            const double inflow = std::max(-mass_flux, 0.0);
            equation.A_p(i, j) += outflow;
            if (isCoupledCell(mesh, face.neighbour_i, face.neighbour_j)) {
                neighbourCoefficient(
                    equation, static_cast<Direction>(face_index), i, j) += inflow;
            } else {
                addBoundaryConvection(
                    mesh, face, i, j, inflow,
                    equation.source[n], source_v[n]);
            }
        }
    }
}

void addLaplacian(
    Mesh& mesh,
    Equation& equation,
    Eigen::VectorXd& source_v,
    double dynamic_viscosity,
    LaplacianScheme scheme)
{
    requireImplemented(
        scheme == LaplacianScheme::Orthogonal,
        "尚未实现指定的 Laplacian 格式");
    requirePositiveFinite(dynamic_viscosity, "动力粘度 ");
    for (int n = 0; n < mesh.internumber; ++n) {
        const int i = mesh.interi[static_cast<std::size_t>(n)];
        const int j = mesh.interj[static_cast<std::size_t>(n)];
        const auto faces = cellFaces(mesh, i, j);
        for (std::size_t face_index = 0; face_index < faces.size(); ++face_index) {
            const FaceData& face = faces[face_index];
            const double diffusion =
                face.area * dynamic_viscosity / face.distance;
            if (isCoupledCell(mesh, face.neighbour_i, face.neighbour_j)) {
                equation.A_p(i, j) += diffusion;
                neighbourCoefficient(
                    equation, static_cast<Direction>(face_index), i, j) +=
                    diffusion;
            } else {
                addBoundaryDiffusion(
                    mesh, face, i, j, diffusion, equation.A_p(i, j),
                    equation.source[n], source_v[n]);
            }
        }
    }
}

void addGradient(
    Mesh& mesh,
    Equation& equation,
    Eigen::VectorXd& source_v,
    GradientScheme scheme)
{
    requireImplemented(
        scheme == GradientScheme::Central,
        "尚未实现指定的压力梯度格式");
    for (int n = 0; n < mesh.internumber; ++n) {
        const int i = mesh.interi[static_cast<std::size_t>(n)];
        const int j = mesh.interj[static_cast<std::size_t>(n)];
        const auto faces = cellFaces(mesh, i, j);
        const double p_w = stencil::pressureSample(mesh, i, j - 1, i, j);
        const double p_e = stencil::pressureSample(mesh, i, j + 1, i, j);
        const double p_n = stencil::pressureSample(mesh, i - 1, j, i, j);
        const double p_s = stencil::pressureSample(mesh, i + 1, j, i, j);
        equation.source[n] +=
            (p_w - p_e) * mesh.vol(i, j) /
            (faces[1].distance + faces[0].distance);
        source_v[n] +=
            (p_n - p_s) * mesh.vol(i, j) /
            (faces[2].distance + faces[3].distance);
    }
}

void applyVelocityEquationRelaxation(
    Mesh& mesh,
    Equation& equation,
    Eigen::VectorXd& source_v,
    double relaxation)
{
    if (!(relaxation > 0.0 && relaxation <= 1.0)) {
        throw std::invalid_argument("速度松弛因子必须在 (0, 1] 内");
    }
    for (int n = 0; n < mesh.internumber; ++n) {
        const int i = mesh.interi[static_cast<std::size_t>(n)];
        const int j = mesh.interj[static_cast<std::size_t>(n)];
        equation.A_e(i, j) *= relaxation;
        equation.A_w(i, j) *= relaxation;
        equation.A_n(i, j) *= relaxation;
        equation.A_s(i, j) *= relaxation;
        equation.source[n] =
            relaxation * equation.source[n] +
            (1.0 - relaxation) * equation.A_p(i, j) * mesh.u_star(i, j);
        source_v[n] =
            relaxation * source_v[n] +
            (1.0 - relaxation) * equation.A_p(i, j) * mesh.v_star(i, j);
    }
}
