#pragma once

#include "numerics/equation.h"
#include "numerics/schemes.h"

void interpolateFaceVelocity(
    Mesh& mesh,
    const Equation& momentum,
    InterpolationScheme scheme);
