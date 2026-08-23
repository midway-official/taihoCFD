#pragma once

#include "numerics/equation.h"
#include "numerics/schemes.h"
#include "numerics/time_term.h"

void addDdt(
    Mesh& mesh,
    Equation& equation,
    Eigen::VectorXd& source_v,
    const TimeTerm& time_term,
    TimeScheme scheme);

void addConvection(
    Mesh& mesh,
    Equation& equation,
    Eigen::VectorXd& source_v,
    ConvectionScheme scheme);

void addLaplacian(
    Mesh& mesh,
    Equation& equation,
    Eigen::VectorXd& source_v,
    double viscosity,
    LaplacianScheme scheme);

void addPressureGradient(
    Mesh& mesh,
    Equation& equation,
    Eigen::VectorXd& source_v,
    GradientScheme scheme);

void applyVelocityEquationRelaxation(
    Mesh& mesh,
    Equation& equation,
    Eigen::VectorXd& source_v,
    double relaxation);
