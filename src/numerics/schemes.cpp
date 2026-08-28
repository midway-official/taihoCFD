#include "numerics/schemes.h"

#include <stdexcept>

NumericalSchemes NumericalSchemes::steady() {
    return {
        TimeScheme::SteadyState,
        ConvectionScheme::Upwind,
        GradientScheme::Central,
        LaplacianScheme::Orthogonal,
        InterpolationScheme::Linear,
    };
}

NumericalSchemes NumericalSchemes::backwardEuler() {
    NumericalSchemes schemes = steady();
    schemes.time = TimeScheme::BackwardEuler;
    return schemes;
}

void NumericalSchemes::validate() const {
    if (time != TimeScheme::SteadyState &&
        time != TimeScheme::BackwardEuler) {
        throw std::invalid_argument("不支持的时间离散格式");
    }
    if (velocity_convection != ConvectionScheme::Upwind ||
        pressure_gradient != GradientScheme::Central ||
        velocity_laplacian != LaplacianScheme::Orthogonal ||
        face_interpolation != InterpolationScheme::Linear) {
        throw std::invalid_argument("NumericalSchemes 包含尚未实现的格式");
    }
}

std::string_view toString(TimeScheme scheme) {
    switch (scheme) {
        case TimeScheme::Unset:
            return "unset";
        case TimeScheme::SteadyState:
            return "steadyState";
        case TimeScheme::BackwardEuler:
            return "backwardEuler";
    }
    return "unknown";
}

std::string_view toString(ConvectionScheme scheme) {
    return scheme == ConvectionScheme::Upwind ? "upwind" :
        (scheme == ConvectionScheme::Unset ? "unset" : "unknown");
}

std::string_view toString(GradientScheme scheme) {
    return scheme == GradientScheme::Central ? "central" :
        (scheme == GradientScheme::Unset ? "unset" : "unknown");
}

std::string_view toString(LaplacianScheme scheme) {
    return scheme == LaplacianScheme::Orthogonal ? "orthogonal" :
        (scheme == LaplacianScheme::Unset ? "unset" : "unknown");
}

std::string_view toString(InterpolationScheme scheme) {
    return scheme == InterpolationScheme::Linear ? "linear" :
        (scheme == InterpolationScheme::Unset ? "unset" : "unknown");
}
