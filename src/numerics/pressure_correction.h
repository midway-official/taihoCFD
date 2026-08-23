#pragma once

#include "numerics/equation.h"

void assemblePressureCorrection(
    Mesh& mesh,
    Equation& pressure,
    const Equation& momentum,
    int rank,
    int num_procs);

void correctPressure(Mesh& mesh, double pressure_relaxation);
void correctVelocity(Mesh& mesh, const Equation& momentum);
