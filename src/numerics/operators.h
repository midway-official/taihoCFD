#pragma once

#include "numerics/equation.h"
#include "numerics/schemes.h"
#include "numerics/time_term.h"

void addDdt(
    Mesh& mesh,
    Equation& equation,
    Eigen::VectorXd& source_v,
    const TimeTerm& time_term,
    double rho,
    TimeScheme scheme);

void addConvection(
    Mesh& mesh,
    Equation& equation,
    Eigen::VectorXd& source_v,
    double rho,
    ConvectionScheme scheme);

void addLaplacian(
    Mesh& mesh,
    Equation& equation,
    Eigen::VectorXd& source_v,
    double dynamic_viscosity,
    LaplacianScheme scheme);

void addGradient(
    Mesh& mesh,
    Equation& equation,
    Eigen::VectorXd& source_v,
    GradientScheme scheme);

void applyVelocityEquationRelaxation(
    Mesh& mesh,
    Equation& equation,
    Eigen::VectorXd& source_v,
    double relaxation);
