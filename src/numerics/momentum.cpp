#include "numerics/momentum.h"

#include "numerics/operators.h"
#include "numerics/stencil.h"

void assembleMomentum(
    Mesh& mesh,
    Equation& momentum,
    Eigen::VectorXd& source_v,
    const FluidProperties& fluid,
    double velocity_relaxation,
    const TimeTerm& time_term,
    const NumericalSchemes& schemes)
{
    schemes.validate();
    fluid.validate();
    momentum.reset();
    source_v.setZero(mesh.internumber);

    addDdt(mesh, momentum, source_v, time_term, fluid.rho, schemes.time);
    addConvection(
        mesh, momentum, source_v, fluid.rho, schemes.velocity_convection);
    addLaplacian(
        mesh, momentum, source_v, fluid.mu,
        schemes.velocity_laplacian);
    addGradient(
        mesh, momentum, source_v, schemes.pressure_gradient);
    applyVelocityEquationRelaxation(
        mesh, momentum, source_v, velocity_relaxation);

    for (int n = 0; n < mesh.internumber; ++n) {
        const int i = mesh.interi[static_cast<std::size_t>(n)];
        const int j = mesh.interj[static_cast<std::size_t>(n)];
        stencil::requireFinite(momentum.A_p(i, j), "动量对角系数", i, j);
        stencil::requireFinite(momentum.source[n], "u 动量源项", i, j);
        stencil::requireFinite(source_v[n], "v 动量源项", i, j);
    }
    momentum.buildMatrix();
}
