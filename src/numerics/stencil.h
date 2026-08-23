#pragma once

#include "numerics/equation.h"

namespace stencil {

double checkedDistance(double value, const char* name, int i, int j);
double pressureSample(
    const Mesh& mesh, int i, int j, int owner_i, int owner_j);
double pressureCorrectionSample(
    const Mesh& mesh, int i, int j, int owner_i, int owner_j);
double xFaceMobility(
    const Mesh& mesh, const Equation& momentum, int i, int j);
double yFaceMobility(
    const Mesh& mesh, const Equation& momentum, int i, int j);
double pressureBoundaryMobility(
    const Mesh& mesh,
    const Equation& momentum,
    int i,
    int j,
    double distance);
void requireFinite(double value, const char* label, int i, int j);

}  // namespace stencil
