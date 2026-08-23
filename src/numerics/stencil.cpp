#include "numerics/stencil.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace stencil {

double checkedDistance(double value, const char* name, int i, int j) {
    if (!std::isfinite(value) || value <= 0.0) {
        std::ostringstream message;
        message << "非法网格距离 " << name << " at (" << i << ", " << j
                << "): " << value;
        throw std::runtime_error(message.str());
    }
    return value;
}

double pressureSample(
    const Mesh& mesh,
    int i,
    int j,
    int owner_i,
    int owner_j)
{
    if (isCoupledCell(mesh, i, j)) {
        return mesh.p(i, j);
    }
    return evaluatePressureBoundary(
        boundaryPatch(mesh, i, j), mesh.p(owner_i, owner_j));
}

double pressureCorrectionSample(
    const Mesh& mesh,
    int i,
    int j,
    int owner_i,
    int owner_j)
{
    if (isCoupledCell(mesh, i, j)) {
        return mesh.p_prime(i, j);
    }
    const ScalarBoundaryCondition& condition =
        boundaryPatch(mesh, i, j).pressure;
    if (fixesValue(condition.type)) {
        return 0.0;
    }
    return mesh.p_prime(owner_i, owner_j);
}

double xFaceMobility(
    const Mesh& mesh,
    const Equation& momentum,
    int i,
    int j)
{
    const double distance = checkedDistance(
        mesh.x_c(i, j + 1) - mesh.x_c(i, j), "dx-face", i, j);
    const double left = mesh.vol(i, j) / momentum.A_p(i, j);
    const double right = mesh.vol(i, j + 1) / momentum.A_p(i, j + 1);
    return 0.5 * (left + right) / distance;
}

double yFaceMobility(
    const Mesh& mesh,
    const Equation& momentum,
    int i,
    int j)
{
    const double distance = checkedDistance(
        mesh.y_c(i + 1, j) - mesh.y_c(i, j), "dy-face", i, j);
    const double lower = mesh.vol(i, j) / momentum.A_p(i, j);
    const double upper = mesh.vol(i + 1, j) / momentum.A_p(i + 1, j);
    return 0.5 * (lower + upper) / distance;
}

double pressureBoundaryMobility(
    const Mesh& mesh,
    const Equation& momentum,
    int i,
    int j,
    double distance)
{
    return (mesh.vol(i, j) / momentum.A_p(i, j)) /
        checkedDistance(distance, "pressure boundary", i, j);
}

void requireFinite(double value, const char* label, int i, int j) {
    if (!std::isfinite(value)) {
        std::ostringstream message;
        message << label << " 产生非有限值 at (" << i << ", " << j << ')';
        throw std::runtime_error(message.str());
    }
}

}  // namespace stencil
