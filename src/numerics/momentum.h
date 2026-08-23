#pragma once

#include "numerics/equation.h"

struct TimeTerm {
    double dt = 0.0;
    const Eigen::MatrixXd* u_previous = nullptr;
    const Eigen::MatrixXd* v_previous = nullptr;

    static TimeTerm none();
    static TimeTerm backwardEuler(
        double dt,
        const Eigen::MatrixXd& u_previous,
        const Eigen::MatrixXd& v_previous);

    bool enabled() const { return dt > 0.0; }
};

void assembleMomentum(
    Mesh& mesh,
    Equation& momentum,
    Eigen::VectorXd& source_v,
    double viscosity,
    double velocity_relaxation,
    const TimeTerm& time_term);
