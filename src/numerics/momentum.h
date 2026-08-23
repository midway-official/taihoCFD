#pragma once

#include "numerics/equation.h"
#include "numerics/schemes.h"
#include "numerics/time_term.h"

void assembleMomentum(
    Mesh& mesh,
    Equation& momentum,
    Eigen::VectorXd& source_v,
    double viscosity,
    double velocity_relaxation,
    const TimeTerm& time_term,
    const NumericalSchemes& schemes);
