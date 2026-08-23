#pragma once

#include <string_view>

enum class TimeScheme {
    SteadyState,
    BackwardEuler,
};

enum class ConvectionScheme {
    Upwind,
};

enum class GradientScheme {
    Central,
};

enum class LaplacianScheme {
    Orthogonal,
};

enum class InterpolationScheme {
    Linear,
};

struct NumericalSchemes {
    TimeScheme time = TimeScheme::SteadyState;
    ConvectionScheme velocity_convection = ConvectionScheme::Upwind;
    GradientScheme pressure_gradient = GradientScheme::Central;
    LaplacianScheme velocity_laplacian = LaplacianScheme::Orthogonal;
    InterpolationScheme face_interpolation = InterpolationScheme::Linear;

    static NumericalSchemes steady();
    static NumericalSchemes backwardEuler();
    void validate() const;
};

std::string_view toString(TimeScheme scheme);
std::string_view toString(ConvectionScheme scheme);
std::string_view toString(GradientScheme scheme);
std::string_view toString(LaplacianScheme scheme);
std::string_view toString(InterpolationScheme scheme);
