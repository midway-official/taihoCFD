#include "numerics/time_term.h"

#include <cmath>
#include <stdexcept>

TimeTerm TimeTerm::none() {
    return {};
}

TimeTerm TimeTerm::backwardEuler(
    double dt_value,
    const Eigen::MatrixXd& old_u,
    const Eigen::MatrixXd& old_v)
{
    if (!(dt_value > 0.0) || !std::isfinite(dt_value)) {
        throw std::invalid_argument("时间步长必须为正有限数");
    }
    return {dt_value, &old_u, &old_v};
}
