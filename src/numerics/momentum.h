#pragma once

#include "numerics/equation.h"
#include "numerics/fluid_properties.h"
#include "numerics/schemes.h"
#include "numerics/time_term.h"

void assembleMomentum(
    Mesh& mesh,
    Equation& momentum,
    Eigen::VectorXd& source_v,
    const FluidProperties& fluid,
    double velocity_relaxation,
    const TimeTerm& time_term,
    const NumericalSchemes& schemes);
