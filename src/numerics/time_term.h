#pragma once

#include <eigen3/Eigen/Core>

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
