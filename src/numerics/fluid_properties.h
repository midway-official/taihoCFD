#pragma once

#include <cmath>
#include <stdexcept>

// Constant transport properties for the current incompressible solver.
struct FluidProperties {
    double rho = 0.0;
    double mu = 0.0;

    void validate() const {
        if (!(rho > 0.0) || !(mu > 0.0) ||
            !std::isfinite(rho) || !std::isfinite(mu)) {
            throw std::invalid_argument("rho 和动力粘度 mu 必须为正有限数");
        }
    }
};
